#include "BitwidthOptimizationsTarget.h"

#include "BitwidthTypeSystem.h"
#include "TargetUtils.h"
#include "hls-fuzzer/BasicCGenerator.h"
#include "hls-fuzzer/TargetRegistry.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/Program.h"

namespace {
REGISTER_TARGET("bitwidth-optimizations",
                dynamatic::BitwidthOptimizationsTarget);
} // namespace

using namespace dynamatic;

namespace {
class BitwidthOptimizationsGenerator : public AbstractWorker {
public:
  explicit BitwidthOptimizationsGenerator(const Options &options,
                                          Randomly &&random)
      : AbstractWorker(options, std::move(random)) {}

  void generate(llvm::raw_ostream &os, llvm::StringRef functionName) override;

  VerificationResult
  verify(const std::filesystem::path &sourceFile) const override;

private:
  std::uint8_t maxBitwidth;
};

} // namespace

std::unique_ptr<AbstractWorker>
BitwidthOptimizationsTarget::createWorker(const Options &options,
                                          Randomly randomly) const {
  return std::make_unique<BitwidthOptimizationsGenerator>(options,
                                                          std::move(randomly));
}

void BitwidthOptimizationsGenerator::generate(llvm::raw_ostream &os,
                                              llvm::StringRef functionName) {
  // Enforce a strict bitwidth requirement for the entire program.
  maxBitwidth = random.getInteger<std::uint8_t>(1, 32);
  gen::BitwidthTypeSystem bitwidthTypeSystem(maxBitwidth, random);
  gen::BasicCGenerator generator(random, bitwidthTypeSystem,
                                 /*entryContext=*/
                                 gen::BitwidthTypingContext{maxBitwidth});

  ast::Function function = generator.generate(functionName);
  os << R"(
#include <stdint.h>
#include <math.h>
#include "dynamatic/Integration.h"

)";
  os << function << '\n';
  os << generator.generateTestBench(function);
}

constexpr std::string_view ORACLE_EXECUTABLE = "hls-fuzzer-check-bitwidth";
constexpr std::string_view COMPILATION_IR_OUTPUT =
    "./out/comp/handshake_export.mlir";

AbstractWorker::VerificationResult BitwidthOptimizationsGenerator::verify(
    const std::filesystem::path &sourceFile) const {
  switch (options.kind) {
  case OracleKind::Functional:
    return performDifferentialTesting(sourceFile,
                                      options.dynamaticExecutablePath);
  case OracleKind::NonFunctional:
    return performNonFunctionalTesting(
        sourceFile, options.dynamaticExecutablePath,
        (std::filesystem::path(options.executablePath).parent_path() /
         ORACLE_EXECUTABLE)
            .string(),
        {COMPILATION_IR_OUTPUT, std::to_string(maxBitwidth)});
  }
  llvm_unreachable("all enum cases handled");
}
