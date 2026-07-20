#include <atomic>
#include <cassert>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <thread>
#include <vector>

#include "Options.h"
#include "OptionsParser.h"
#include "TargetRegistry.h"

#include "llvm/DebugInfo/LogicalView/Core/LVOptions.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"

static std::atomic_bool quit = false;
static std::mutex errorMutex;
// Serializes the sampling and writing of a JSON report, which any worker thread
// may do once it has verified a program. Without it two reporters could rename
// their snapshots in the opposite order in which they took them, leaving the
// older of the two behind as the final content of the file.
static std::mutex reportMutex;
static std::atomic_uint64_t testCaseCounter = 0;
static std::atomic_uint64_t bugCounter = 0;

static bool hasProgramLimit = false;
// Number of programs left to generate and verify, decremented by each
// worker thread as it claims a program to work on. Only meaningful if
// 'hasProgramLimit' is set; may go negative once the limit has been
// reached, since every worker that fails to claim a program still performs
// the decrement.
static std::atomic_int64_t remainingPrograms = 0;

namespace {
/// Everything required to report on the progress of a fuzzing run. Shared by
/// every thread and immutable once all workers have been created.
struct ReportConfig {
  const dynamatic::Options &options;
  const std::vector<std::unique_ptr<dynamatic::AbstractWorker>> &workers;
  std::chrono::high_resolution_clock::time_point startTime;
};
} // namespace

static void reportJSON(const ReportConfig &config);

static void threadWork(dynamatic::AbstractWorker &target,
                       const std::filesystem::path &workingDirectory,
                       const std::string &functionName,
                       const ReportConfig &config) {
  while (!quit) {
    if (hasProgramLimit && remainingPrograms.fetch_sub(1) <= 0) {
      quit = true;
      break;
    }

    std::filesystem::remove_all(workingDirectory);
    std::filesystem::create_directories(workingDirectory);
    std::filesystem::path sourceFile = workingDirectory / (functionName + ".c");

    llvm::cantFail(
        llvm::writeToOutput(sourceFile.string(), [&](llvm::raw_ostream &os) {
          target.generate(os, functionName);
          return llvm::Error::success();
        }));

    dynamatic::AbstractWorker::VerificationResult result =
        target.verify(sourceFile);
    ++testCaseCounter;
    switch (result) {
    case dynamatic::AbstractWorker::Success:
      break;
    case dynamatic::AbstractWorker::Bug:
      ++bugCounter;
      std::filesystem::path destination;
      std::size_t i = 0;
      while (true) {
        destination = workingDirectory / ".." / ("bug" + std::to_string(i));
        if (std::filesystem::create_directory(destination))
          break;

        i++;
      }

      std::filesystem::copy(
          workingDirectory, destination,
          std::filesystem::copy_options::recursive |
              std::filesystem::copy_options::overwrite_existing);
      {
        std::scoped_lock<std::mutex> lock{errorMutex};
        std::cerr << destination << " written" << std::endl;
      }
      break;
    }

    // Report once the outcome of the program is known, so that the file
    // changes exactly when the data it holds does. Verifying a program takes
    // seconds at minimum, making this no more frequent than the heartbeat the
    // console report uses.
    if (config.options.jsonOutput)
      reportJSON(config);
  }
}

/// Collects the statistics of all 'workers', combining objects of the same
/// category into a single one.
static std::map<std::string, dynamatic::Statistic> mergeStatistics(
    const std::vector<std::unique_ptr<dynamatic::AbstractWorker>> &workers) {
  std::map<std::string, dynamatic::Statistic> merged;
  for (const std::unique_ptr<dynamatic::AbstractWorker> &worker : workers) {
    for (dynamatic::Statistic &workerStats : worker->getStatistics()) {
      auto [iter, inserted] = merged.try_emplace(
          workerStats.getCategory().str(), std::move(workerStats));
      // Initial insertion into the map, nothing to do.
      if (inserted)
        continue;

      iter->second.merge(workerStats);
    }
  }
  return merged;
}

namespace {
/// Snapshot of the progress of a fuzzing run at a point in time.
struct Progress {
  std::uint64_t numPrograms;
  std::uint64_t numBugs;
  std::size_t seconds;
  double testsPerSecond;
  std::map<std::string, dynamatic::Statistic> statistics;
};
} // namespace

/// Samples the progress made by the fuzzing run described by 'config' so far.
static Progress sampleProgress(const ReportConfig &config) {
  Progress progress;
  progress.numPrograms = testCaseCounter.load();
  progress.numBugs = bugCounter.load();
  progress.seconds =
      std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::high_resolution_clock::now() - config.startTime)
          .count();
  progress.testsPerSecond = 0.0;
  if (progress.seconds != 0)
    progress.testsPerSecond = static_cast<double>(progress.numPrograms) /
                              static_cast<double>(progress.seconds);

  if (config.options.statistics)
    progress.statistics = mergeStatistics(config.workers);

  return progress;
}

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
      {"testsPerSecond", progress.testsPerSecond},
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

/// Writes the current fuzzing progress to the file named by '--json-output'.
/// May be called from any thread; the progress is sampled under the same lock
/// that covers the write, so that the file always ends up holding the newest of
/// any two concurrently taken samples.
static void reportJSON(const ReportConfig &config) {
  assert(config.options.jsonOutput && "'--json-output' was not requested");

  std::scoped_lock<std::mutex> lock{reportMutex};
  Progress progress = sampleProgress(config);
  if (llvm::Error error =
          writeJSONReport(*config.options.jsonOutput, progress)) {
    std::scoped_lock<std::mutex> errorLock{errorMutex};
    llvm::errs() << "Failed to write '" << *config.options.jsonOutput
                 << "': " << llvm::toString(std::move(error)) << '\n';
  }
}

/// Prints the current fuzzing progress, along with the statistics gathered by
/// the workers if any were requested, to the console.
static void reportConsole(const ReportConfig &config) {
  Progress progress = sampleProgress(config);

  std::scoped_lock<std::mutex> lock{errorMutex};
  std::cerr << "Current test rate: " << progress.testsPerSecond
            << " tests per second [" << progress.numPrograms << '/'
            << progress.seconds << "s]; " << progress.numBugs << " bugs found"
            << std::endl;
  for (auto &&[category, statistic] : progress.statistics)
    statistic.print(llvm::errs());
}

int main(int argc, char **argv) {
  auto signalHandler = +[](int) { quit = true; };

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

  std::optional<std::size_t> numThreads = optionsParser.getNumThreads();
  if (!numThreads)
    return -1;

  if (std::optional<std::size_t> numPrograms = optionsParser.getNumPrograms()) {
    hasProgramLimit = true;
    remainingPrograms = static_cast<std::int64_t>(*numPrograms);
  }

  // Every worker reports on the statistics of all the others, so they must all
  // exist before any of the threads below start reading them.
  std::vector<std::unique_ptr<dynamatic::AbstractWorker>> workers(*numThreads);
  for (std::unique_ptr<dynamatic::AbstractWorker> &worker : workers) {
    size_t seed = std::random_device()();
    {
      std::scoped_lock<std::mutex> lock{errorMutex};
      std::cerr << "Using seed: " << seed << '\n';
    }
    worker = target->createWorker(options, dynamatic::Randomly(seed));
  }

  const ReportConfig config{options, workers,
                            std::chrono::high_resolution_clock::now()};

  std::vector<std::thread> threads(*numThreads);
  for (size_t i = 0; i < threads.size(); i++) {
    std::filesystem::path workingDirectory =
        std::filesystem::current_path() / ("thread" + std::to_string(i));
    threads[i] = std::thread(threadWork, std::ref(*workers[i]),
                             std::move(workingDirectory),
                             "test" + std::to_string(i), std::cref(config));
  }

  while (!quit) {
    std::this_thread::sleep_for(std::chrono::seconds(3));
    // The JSON report is driven by the workers themselves, leaving the
    // heartbeat to keep only the console up to date.
    if (!options.jsonOutput)
      reportConsole(config);
  }

  for (auto &iter : threads) {
    iter.join();
  }

  // Report one last time so that a run which never got to verify a program
  // still leaves the file behind, and so that a console run is summarised.
  if (options.jsonOutput)
    reportJSON(config);
  else
    reportConsole(config);
  return 0;
}
