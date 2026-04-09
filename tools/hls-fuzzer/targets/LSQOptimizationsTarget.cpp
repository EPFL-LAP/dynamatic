#include "LSQOptimizationsTarget.h"

#include "LSQNoDepTypeSystem.h"
#include "TargetUtils.h"
#include "hls-fuzzer/BasicCGenerator.h"
#include "hls-fuzzer/TargetRegistry.h"

namespace {
REGISTER_TARGET("lsq-optimizations", dynamatic::LSQOptimizationsTarget);
} // namespace

using namespace dynamatic;

namespace {
class LSQOptimizationsWorker : public AbstractWorker {
public:
  explicit LSQOptimizationsWorker(const Options &options, Randomly &&random)
      : AbstractWorker(options, std::move(random)) {}

  void generate(llvm::raw_ostream &os, llvm::StringRef functionName) override;

  VerificationResult
  verify(const std::filesystem::path &sourceFile) const override;
};

} // namespace

std::unique_ptr<AbstractWorker>
LSQOptimizationsTarget::createWorker(const Options &options,
                                     Randomly randomly) const {
  return std::make_unique<LSQOptimizationsWorker>(options, std::move(randomly));
}

void LSQOptimizationsWorker::generate(llvm::raw_ostream &os,
                                      llvm::StringRef functionName) {
  gen::LSQNoDepTypeSystem typeSystem(random);
  gen::BasicCGenerator generator(
      random, typeSystem,
      /*entryContext=*/
      {gen::LSQNoDepContext{},
        // Randomly decide
       {random.fromEnum<gen::DynamaticTypingContext::Constraint>()}});

  ast::Function function = generator.generate(functionName);
  os << R"(
#include <stdint.h>
#include <math.h>
#include "dynamatic/Integration.h"

)";
  os << function << '\n';
  os << generator.generateTestBench(function);
}

constexpr std::string_view ORACLE_EXECUTABLE = "hls-fuzzer-check-no-lsq";
constexpr std::string_view COMPILATION_IR_OUTPUT =
    "./out/comp/handshake_export.mlir";

AbstractWorker::VerificationResult
LSQOptimizationsWorker::verify(const std::filesystem::path &sourceFile) const {
  return performNonFunctionalTesting(
      sourceFile, options.dynamaticExecutablePath,
      (std::filesystem::path(options.executablePath).parent_path() /
       ORACLE_EXECUTABLE)
          .string(),
      {COMPILATION_IR_OUTPUT});
}
