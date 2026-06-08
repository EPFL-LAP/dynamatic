#include <atomic>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "Options.h"
#include "OptionsParser.h"
#include "TargetRegistry.h"

#include "llvm/DebugInfo/LogicalView/Core/LVOptions.h"
#include "llvm/Support/FileSystem.h"

static std::atomic_bool quit = false;
static std::mutex errorMutex;
static std::atomic_uint64_t testCaseCounter = 0;
static std::atomic_uint64_t bugCounter = 0;

static void threadWork(const std::unique_ptr<dynamatic::AbstractWorker> &target,
                       const std::filesystem::path &workingDirectory,
                       const std::string &functionName) {
  while (!quit) {
    std::filesystem::remove_all(workingDirectory);
    std::filesystem::create_directories(workingDirectory);
    std::filesystem::path sourceFile = workingDirectory / (functionName + ".c");

    llvm::cantFail(
        llvm::writeToOutput(sourceFile.string(), [&](llvm::raw_ostream &os) {
          target->generate(os, functionName);
          return llvm::Error::success();
        }));

    dynamatic::AbstractWorker::VerificationResult result =
        target->verify(sourceFile);
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
  }
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

  std::vector<std::thread> threads(*numThreads);
  for (size_t i = 0; i < threads.size(); i++) {
    size_t seed = std::random_device()();
    {
      std::scoped_lock<std::mutex> lock{errorMutex};
      std::cerr << "Using seed: " << seed << '\n';
    }

    std::filesystem::path workingDirectory =
        std::filesystem::current_path() / ("thread" + std::to_string(i));
    threads[i] = std::thread(
        threadWork, target->createWorker(options, dynamatic::Randomly(seed)),
        std::move(workingDirectory), "test" + std::to_string(i));
  }

  auto startTime = std::chrono::high_resolution_clock::now();
  while (!quit) {
    std::this_thread::sleep_for(std::chrono::seconds(3));
    uint64_t counter = testCaseCounter.load();
    std::size_t seconds =
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::high_resolution_clock::now() - startTime)
            .count();
    double rate = static_cast<double>(counter) / static_cast<double>(seconds);
    {
      std::lock_guard<std::mutex> lock{errorMutex};
      std::cerr << "Current test rate: " << rate << " tests per second ["
                << testCaseCounter << '/' << seconds << "s]; "
                << bugCounter.load() << " bugs found" << std::endl;
    }
  }

  for (auto &iter : threads) {
    iter.join();
  }

  return 0;
}
