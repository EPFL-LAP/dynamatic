#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "Options.h"
#include "OptionsParser.h"
#include "TargetRegistry.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"

// Whether fuzzing should stop, either because the run was interrupted or
// because all the programs that were asked for have been generated.
static std::atomic_bool quit = false;
// Whether the run was interrupted by a signal. Kept apart from 'quit' since the
// signal is delivered to the whole process group, taking down the programs that
// are currently being worked on as well, while reaching the program limit
// leaves them to finish.
static std::atomic_bool interrupted = false;
static std::mutex errorMutex;

static bool hasProgramLimit = false;
// Number of programs left to generate and verify, decremented by each worker
// thread as it claims a program to work on. Only meaningful if
// 'hasProgramLimit' is set; may go negative once the limit has been reached,
// since every worker that fails to claim a program still performs the
// decrement.
static std::atomic_int64_t remainingPrograms = 0;

namespace {
/// Snapshot of the progress made at a point in time.
struct Progress {
  /// Number of programs that were generated and verified.
  std::uint64_t numPrograms = 0;
  /// Number of the above programs that turned out to trigger a bug.
  std::uint64_t numBugs = 0;
  /// Wall clock time that was spent doing so.
  std::chrono::seconds duration{0};
  /// Statistics gathered along the way, one per category.
  std::map<std::string, dynamatic::Statistic> statistics;

  /// Returns the rate at which programs were verified.
  double getTestsPerSecond() const {
    if (duration.count() == 0)
      return 0.0;

    return static_cast<double>(numPrograms) /
           static_cast<double>(duration.count());
  }

  /// Merges 'statistic' into the statistics of its category.
  void merge(const dynamatic::Statistic &statistic) {
    auto [iter, inserted] =
        statistics.try_emplace(statistic.getCategory().str(), statistic);
    // Initial insertion into the map, nothing to do.
    if (inserted)
      return;

    iter->second.merge(statistic);
  }

  /// Adds the programs, bugs and statistics of 'other' to this progress.
  /// 'duration' is not additive and is left untouched: it is the wall clock
  /// time of the run as a whole, not the sum of the time its programs took.
  void merge(const Progress &other) {
    numPrograms += other.numPrograms;
    numBugs += other.numBugs;
    for (const auto &[category, statistic] : other.statistics)
      merge(statistic);
  }
};
} // namespace

/// Writes 'progress' to 'file' as JSON.
///
/// The report is first written to a temporary file next to 'file' and only then
/// atomically renamed onto it. A crash at any point during the write therefore
/// leaves 'file' containing the previous report rather than a half-written one.
static llvm::Error writeJSONReport(llvm::StringRef file,
                                   const Progress &progress) {
  llvm::json::Object jsonStatistics;
  for (const auto &[category, statistic] : progress.statistics)
    jsonStatistics[category] = statistic.toJSON();

  llvm::json::Value report = llvm::json::Object{
      {"numPrograms", progress.numPrograms},
      {"numBugs", progress.numBugs},
      {"testsPerSecond", progress.getTestsPerSecond()},
      {"statistics", std::move(jsonStatistics)},
  };

  // The temporary must be created next to 'file' rather than in the system
  // temporary directory, as a rename is only atomic within one filesystem.
  llvm::Expected<llvm::sys::fs::TempFile> temp =
      llvm::sys::fs::TempFile::create(file + ".%%%%%%%%.tmp");
  if (!temp)
    return temp.takeError();

  llvm::raw_fd_ostream os(temp->FD, /*shouldClose=*/false);
  llvm::json::OStream(os, /*IndentSize=*/2).value(report);
  os << '\n';
  os.flush();
  if (os.has_error()) {
    std::error_code error = os.error();
    os.clear_error();
    llvm::consumeError(temp->discard());
    return llvm::errorCodeToError(error);
  }

  // 'keep' removes the temporary itself if the rename fails.
  return temp->keep(file);
}

/// Writes 'progress' to 'file' as JSON, reporting a failure to do so on the
/// console.
static void reportJSON(llvm::StringRef file, const Progress &progress) {
  if (llvm::Error error = writeJSONReport(file, progress)) {
    std::scoped_lock<std::mutex> lock{errorMutex};
    llvm::errs() << "Failed to write '" << file
                 << "': " << llvm::toString(std::move(error)) << '\n';
  }
}

/// Prints 'progress', along with the statistics gathered by the workers if any
/// were requested, to the console.
static void reportConsole(const Progress &progress) {
  std::scoped_lock<std::mutex> lock{errorMutex};
  std::cerr << "Current test rate: " << progress.getTestsPerSecond()
            << " tests per second [" << progress.numPrograms << '/'
            << progress.duration.count() << "s]; " << progress.numBugs
            << " bugs found" << std::endl;
  for (auto &&[category, statistic] : progress.statistics)
    statistic.print(llvm::errs());
}

// Progress made by this process so far. Guarded by 'progressMutex', which also
// covers the reports written from it, so that the report file always ends up
// holding the newest of any two concurrently taken samples.
static std::mutex progressMutex;
static Progress totalProgress;
static const std::chrono::high_resolution_clock::time_point startTime =
    std::chrono::high_resolution_clock::now();

/// Adds the progress 'made' to the progress of this process. May be called from
/// any thread.
static void addProgress(const Progress &made) {
  std::scoped_lock<std::mutex> lock{progressMutex};
  totalProgress.merge(made);
}

/// Reports the progress made by this process so far, as JSON or on the console
/// depending on 'options'. May be called from any thread.
static void report(const dynamatic::Options &options) {
  std::scoped_lock<std::mutex> lock{progressMutex};
  totalProgress.duration = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::high_resolution_clock::now() - startTime);

  if (options.jsonOutput)
    reportJSON(*options.jsonOutput, totalProgress);
  else
    reportConsole(totalProgress);
}

/// Generates a program into 'directory' and verifies it, returning what doing
/// so amounted to.
///
/// 'directory' is wiped beforehand and serves as scratch space for the
/// verification, leaving whatever it contains afterwards the reproducer of any
/// bug that was found.
static Progress runSingleProgram(const dynamatic::AbstractTarget &target,
                                 const dynamatic::Options &options,
                                 const std::filesystem::path &directory) {
  std::unique_ptr<dynamatic::AbstractWorker> worker =
      target.createWorker(options, dynamatic::Randomly(std::random_device()()));

  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);

  const std::string functionName = "test";
  std::filesystem::path sourceFile = directory / (functionName + ".c");
  llvm::cantFail(
      llvm::writeToOutput(sourceFile.string(), [&](llvm::raw_ostream &os) {
        worker->generate(os, functionName);
        return llvm::Error::success();
      }));

  Progress progress;
  progress.numPrograms = 1;
  progress.numBugs =
      worker->verify(sourceFile) == dynamatic::AbstractWorker::Bug;
  for (const dynamatic::Statistic &statistic : worker->getStatistics())
    progress.merge(statistic);

  return progress;
}

/// Returns one empty statistic per category that the workers of 'target'
/// gather, or none if '--statistics' was not given. This is required since we
/// need to have a specific instance of 'dynamatic::Statistic' to parse a JSON
/// into.
static std::map<std::string, dynamatic::Statistic>
createStatisticPrototypes(const dynamatic::AbstractTarget &target,
                          const dynamatic::Options &options) {
  std::map<std::string, dynamatic::Statistic> prototypes;
  if (!options.statistics)
    return prototypes;

  std::unique_ptr<dynamatic::AbstractWorker> worker =
      target.createWorker(options, dynamatic::Randomly(std::random_device()()));
  for (dynamatic::Statistic &statistic : worker->getStatistics())
    prototypes.try_emplace(statistic.getCategory().str(), std::move(statistic));

  return prototypes;
}

/// Parses the 'progress' another process reported as 'value'. 'prototypes'
/// provides an empty statistic per category, into which the statistics of the
/// report are parsed.
static bool
parseProgress(const llvm::json::Value &value, Progress &progress,
              const std::map<std::string, dynamatic::Statistic> &prototypes,
              const llvm::json::Path &path) {
  // The test rate the process reported is deliberately not read back, as the
  // rate of a run is not the sum of the rates of its programs. It is recomputed
  // from the start time of this process instead.
  llvm::json::ObjectMapper mapper(value, path);
  if (!mapper || !mapper.map("numPrograms", progress.numPrograms) ||
      !mapper.map("numBugs", progress.numBugs))
    return false;

  // A valid 'mapper' guarantees that 'value' is an object.
  llvm::json::Path statisticsPath = path.field("statistics");
  const llvm::json::Object *statistics =
      value.getAsObject()->getObject("statistics");
  if (!statistics) {
    statisticsPath.report("expected object");
    return false;
  }

  for (const auto &[category, statistic] : *statistics) {
    auto iter = prototypes.find(std::string(llvm::StringRef(category)));
    // A statistic that is of no interest to this run. Cannot occur for a report
    // written by a process this one started, as it passes on its own
    // '--statistics'.
    if (iter == prototypes.end())
      continue;

    dynamatic::Statistic parsed = iter->second;
    if (!parsed.fromJSON(statistic, statisticsPath.field(category)))
      return false;

    progress.statistics.try_emplace(iter->first, std::move(parsed));
  }
  return true;
}

/// Reads the progress another process reported to 'file'. Returns
/// 'std::nullopt' if the report could not be read, in which case the progress
/// it holds is lost and the reason is reported on the console.
static std::optional<Progress>
readReport(const std::filesystem::path &file,
           const std::map<std::string, dynamatic::Statistic> &prototypes) {
  auto fail = [&](llvm::Error error) -> std::optional<Progress> {
    std::scoped_lock<std::mutex> lock{errorMutex};
    llvm::errs() << "Failed to read '" << file.string()
                 << "': " << llvm::toString(std::move(error)) << '\n';
    return std::nullopt;
  };

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(file.string());
  if (std::error_code error = buffer.getError())
    return fail(llvm::errorCodeToError(error));

  llvm::Expected<llvm::json::Value> value =
      llvm::json::parse((*buffer)->getBuffer());
  if (!value)
    return fail(value.takeError());

  Progress progress;
  llvm::json::Path::Root root;
  if (!parseProgress(*value, progress, prototypes, root))
    return fail(root.getError());

  return progress;
}

/// Saves the program contained in 'directory' as a reproducer next to it, so
/// that it survives the next program the worker generates in there.
static void saveReproducer(const std::filesystem::path &directory) {
  std::filesystem::path destination;
  for (std::size_t i = 0;; i++) {
    destination = directory.parent_path() / ("bug" + std::to_string(i));
    // A 'create_directory' that finds the directory already in place returns
    // false rather than failing, which is how a name is claimed atomically
    // against the other workers doing the same.
    if (std::filesystem::create_directory(destination))
      break;
  }

  std::filesystem::copy(directory, destination,
                        std::filesystem::copy_options::recursive |
                            std::filesystem::copy_options::overwrite_existing);

  std::scoped_lock<std::mutex> lock{errorMutex};
  std::cerr << destination << " written" << std::endl;
}

/// Generates and verifies a single program in 'directory' by re-executing this
/// binary as described by 'args' in '--run-as-worker-subprocess' mode. Doing so
/// in a process of its own keeps a crash of the generation or the verification
/// from taking the whole fuzzer down with it.
///
/// Returns the progress that process reported, or a bug of its own if it did
/// not survive its program.
static Progress runProgramProcess(
    const dynamatic::Options &options, llvm::ArrayRef<char *> args,
    const std::filesystem::path &directory,
    const std::map<std::string, dynamatic::Statistic> &prototypes) {
  std::filesystem::path reportFile = directory / "report.json";

  llvm::SmallVector<std::string> storage;
  storage.emplace_back(options.executablePath);
  for (char *arg : args.drop_front())
    storage.emplace_back(arg);

  // All options are appended last, as the last occurrence of an option takes
  // precedence over any the user may have specified themselves.
  storage.emplace_back("--run-as-worker-subprocess");
  storage.emplace_back(directory.string());
  // Have the process report into the program's directory rather than wherever
  // the user asked for the report of the run as a whole to go. This process
  // merges the reports of all its programs into the latter, and the option
  // doubles as keeping the process from reporting on the console itself.
  storage.emplace_back("--json-output");
  storage.emplace_back(reportFile.string());

  llvm::SmallVector<llvm::StringRef> childArgs(storage.begin(), storage.end());

  std::string errorMessage;
  bool executionFailed = false;
  int exitCode = llvm::sys::ExecuteAndWait(
      options.executablePath, childArgs, /*Env=*/std::nullopt,
      /*Redirects=*/{}, /*SecondsToWait=*/0, /*MemoryLimit=*/0, &errorMessage,
      &executionFailed);
  // Failing to start the process at all is a problem with the invocation rather
  // than a crash, making it pointless to try again with the next program.
  if (executionFailed) {
    std::scoped_lock<std::mutex> lock{errorMutex};
    std::cerr << "Failed to start '" << options.executablePath
              << "': " << errorMessage << std::endl;
    quit = true;
    return {};
  }

  if (exitCode == 0)
    return readReport(reportFile, prototypes).value_or(Progress{});

  // The process was taken down along with the rest of the run rather than by
  // the program it was working on, which is therefore not a bug.
  if (interrupted)
    return {};

  {
    std::scoped_lock<std::mutex> lock{errorMutex};
    std::cerr << "Generating or verifying the program in " << directory
              << " exited with code " << exitCode;
    if (!errorMessage.empty())
      std::cerr << " (" << errorMessage << ")";

    std::cerr << std::endl;
  }

  // A crash of the fuzzer or of the tools it drives is a bug just like one an
  // oracle found, and the program that triggered it is worth a reproducer just
  // the same.
  Progress progress;
  progress.numPrograms = 1;
  progress.numBugs = 1;
  return progress;
}

/// Generates and verifies programs in 'directory', one at a time, until fuzzing
/// is done or interrupted. 'runProgram' does a single program and returns what
/// doing so amounted to, either in this process or in one of its own.
static void work(const dynamatic::Options &options,
                 const std::filesystem::path &directory,
                 const std::function<Progress()> &runProgram) {
  while (!quit) {
    if (hasProgramLimit && remainingPrograms.fetch_sub(1) <= 0) {
      quit = true;
      break;
    }

    Progress made = runProgram();
    // Save the reproducer before the next program wipes the directory it is
    // still sitting in.
    if (made.numBugs != 0)
      saveReproducer(directory);

    addProgress(made);
    // Report once the outcome of the program is known, so that the file changes
    // exactly when the data it holds does. Verifying a program takes seconds at
    // minimum, making this no more frequent than the heartbeat the console
    // report uses.
    if (options.jsonOutput)
      report(options);
  }
}

int main(int argc, char **argv) {
  auto signalHandler = +[](int) {
    interrupted = true;
    quit = true;
  };

  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);
  std::signal(SIGSTOP, signalHandler);
  std::signal(SIGPIPE, signalHandler);

  llvm::SmallVector<char *> args(argv, argv + argc);
  auto optionsParser = dynamatic::OptionsParser(args);
  if (optionsParser.shouldDisplayHelp()) {
    optionsParser.printHelp(llvm::outs());
    return 0;
  }

  dynamatic::TargetRegistry &instance =
      dynamatic::TargetRegistry::getInstance();

  std::string targetName = optionsParser.getTargetName();
  if (targetName.empty()) {
    llvm::errs() << "Missing '--target' argument\n";
    llvm::errs() << "Registered targets: ";
    llvm::interleaveComma(instance.listTargets(), llvm::errs());
    return -1;
  }
  std::unique_ptr<dynamatic::AbstractTarget> target =
      instance.getTarget(targetName);
  if (!target) {
    llvm::errs() << "Unknown target '" << targetName << "'\n";
    llvm::errs() << "Registered targets: ";
    llvm::interleaveComma(instance.listTargets(), llvm::errs());
    return -1;
  }

  dynamatic::Options defaults{};
#pragma clang diagnostic ignored "-Wmain"
  defaults.executablePath = llvm::sys::fs::getMainExecutable(
      argv[0], reinterpret_cast<void *>(&main));
  defaults.dynamaticExecutablePath =
      std::filesystem::path(defaults.executablePath).parent_path() /
      "dynamatic";

  auto options = optionsParser.apply(defaults);

  // Work on the single program a fuzzing process asked for and report what that
  // amounted to, which is all the process learns about a program of its own
  // that crashed halfway through.
  if (std::optional<std::string> directory =
          optionsParser.getSingleProgramDirectory()) {
    addProgress(runSingleProgram(*target, options, *directory));
    report(options);
    return 0;
  }

  std::optional<std::size_t> numThreads = optionsParser.getNumThreads();
  if (!numThreads)
    return -1;

  if (std::optional<std::size_t> numPrograms = optionsParser.getNumPrograms()) {
    hasProgramLimit = true;
    remainingPrograms = static_cast<std::int64_t>(*numPrograms);
  }

  bool singleProcess = optionsParser.isSingleProcess();
  // Only ever used to parse the reports of the processes generating the
  // programs, which '--single-process' does without.
  std::map<std::string, dynamatic::Statistic> prototypes;
  if (!singleProcess)
    prototypes = createStatisticPrototypes(*target, options);

  std::vector<std::thread> threads;
  threads.reserve(*numThreads);
  for (std::size_t i = 0; i < *numThreads; i++) {
    std::filesystem::path directory =
        std::filesystem::current_path() / ("thread" + std::to_string(i));

    std::function<Progress()> runProgram;
    if (singleProcess) {
      runProgram = [&target, &options, directory] {
        return runSingleProgram(*target, options, directory);
      };
    } else {
      runProgram = [&options, args = llvm::ArrayRef<char *>(args), directory,
                    &prototypes] {
        return runProgramProcess(options, args, directory, prototypes);
      };
    }

    threads.emplace_back(work, std::cref(options), directory,
                         std::move(runProgram));
  }

  while (!quit) {
    std::this_thread::sleep_for(std::chrono::seconds(3));
    // The JSON report is driven by the workers themselves, leaving the
    // heartbeat to keep only the console up to date.
    if (!options.jsonOutput)
      report(options);
  }

  for (std::thread &thread : threads)
    thread.join();

  // Report one last time so that a run which never got to verify a program
  // still leaves the file behind, and so that a console run is summarised.
  report(options);
  return 0;
}
