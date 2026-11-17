#include "RandomCTarget.h"

#include "RandomCTypeSystem.h"
#include "TargetUtils.h"
#include "hls-fuzzer/BasicCGenerator.h"
#include "hls-fuzzer/TargetRegistry.h"
#include "hls-fuzzer/statistics/ASTStatistic.h"
#include "hls-fuzzer/statistics/IIStatistic.h"

#include <mutex>

REGISTER_TARGET("random-c", dynamatic::RandomCTarget);

using namespace dynamatic;

namespace {
class RandomCWorker : public AbstractWorker {
public:
  explicit RandomCWorker(const Options &options, Randomly &&random)
      : AbstractWorker(options, std::move(random)) {}

  void generate(llvm::raw_ostream &os, llvm::StringRef functionName) override;

  VerificationResult
  verify(const std::filesystem::path &sourceFile) const override;

  std::vector<Statistic> getStatistics() const override {
    std::lock_guard<std::mutex> guard{statisticMutex};
    std::vector<Statistic> stats;
    if (isStatisticEnabled(ASTStatistic::CATEGORY))
      stats.emplace_back(ASTStatistic::CATEGORY.str(), astStatistic);
    if (isStatisticEnabled(IIStatistic::CATEGORY))
      stats.emplace_back(IIStatistic::CATEGORY.str(), iiStatistic);
    return stats;
  }

private:
  mutable std::mutex statisticMutex;
  ASTStatistic astStatistic;
  mutable IIStatistic iiStatistic;
};

} // namespace

std::unique_ptr<AbstractWorker>
RandomCTarget::createWorker(const Options &options, Randomly randomly) const {
  return std::make_unique<RandomCWorker>(options, std::move(randomly));
}

void RandomCWorker::generate(llvm::raw_ostream &os,
                             llvm::StringRef functionName) {
  gen::RandomCTypeSystem dynamaticTypeSystem(random);
  gen::BasicCGenerator generator(random, dynamaticTypeSystem);
  ast::Function function = generator.generate(os, functionName);

  if (isStatisticEnabled(ASTStatistic::CATEGORY)) {
    std::lock_guard<std::mutex> guard{statisticMutex};
    astStatistic.update(function);
  }
}

AbstractWorker::VerificationResult
RandomCWorker::verify(const std::filesystem::path &sourceFile) const {
  bool instrumentII = isStatisticEnabled(IIStatistic::CATEGORY);
  VerificationResult result = performDifferentialTesting(
      sourceFile, options.dynamaticExecutablePath,
      DynamaticOptions().withTimeout(20000).enableII(instrumentII));

  // The simulation artifacts only exist if the flow ran to completion (i.e. no
  // bug was found and the simulation produced a report).
  if (result == Success && instrumentII) {
    std::lock_guard<std::mutex> guard{statisticMutex};
    iiStatistic.update(sourceFile.parent_path() / "out");
  }

  return result;
}
