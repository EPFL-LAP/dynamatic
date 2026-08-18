#include "RandomCTarget.h"

#include "RandomCTypeSystem.h"
#include "TargetUtils.h"
#include "hls-fuzzer/BasicCGenerator.h"
#include "hls-fuzzer/ConjunctionTypeSystem.h"
#include "hls-fuzzer/LimitTypeSystem.h"
#include "hls-fuzzer/TargetRegistry.h"
#include "hls-fuzzer/statistics/ASTStatistic.h"

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
    if (!isStatisticEnabled(ASTStatistic::CATEGORY))
      return {};

    std::lock_guard<std::mutex> guard{statisticMutex};
    return {Statistic(ASTStatistic::CATEGORY.str(), astStatistic)};
  }

private:
  mutable std::mutex statisticMutex;
  ASTStatistic astStatistic;
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
  return performDifferentialTesting(sourceFile, options.dynamaticExecutablePath,
                                    20000);
}
