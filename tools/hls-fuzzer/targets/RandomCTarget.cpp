#include "RandomCTarget.h"

#include "DynamaticTypeSystem.h"
#include "TargetUtils.h"
#include "hls-fuzzer/BasicCGenerator.h"
#include "hls-fuzzer/TargetRegistry.h"

REGISTER_TARGET("random-c", dynamatic::RandomCTarget);

using namespace dynamatic;

namespace {
class RandomCWorker : public AbstractWorker {
public:
  explicit RandomCWorker(const Options &options, Randomly &&random)
      : AbstractWorker(options, std::move(random)) {}

  void generate(llvm::raw_ostream &os,
                llvm::StringRef functionName) override;

  VerificationResult
  verify(const std::filesystem::path &sourceFile) const override;
};

} // namespace

std::unique_ptr<AbstractWorker>
RandomCTarget::createWorker(const Options &options, Randomly randomly) const {
  return std::make_unique<RandomCWorker>(options, std::move(randomly));
}

void RandomCWorker::generate(llvm::raw_ostream &os,
                             llvm::StringRef functionName) {
  gen::DynamaticTypeSystem dynamaticTypeSystem(random);
  gen::BasicCGenerator generator(
      random, dynamaticTypeSystem,
      /*entryContext=*/
      {random.fromEnum<gen::DynamaticTypingContext::Constraint>()});

  ast::Function function = generator.generate(functionName);
  os << R"(
#include <stdint.h>
#include <math.h>
#include "dynamatic/Integration.h"

)";
  os << function << '\n';
  os << generator.generateTestBench(function);
}

AbstractWorker::VerificationResult
RandomCWorker::verify(const std::filesystem::path &sourceFile) const {
  return performDifferentialTesting(sourceFile,
                                    options.dynamaticExecutablePath);
}
