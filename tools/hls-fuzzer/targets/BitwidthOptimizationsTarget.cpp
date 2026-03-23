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

  void generate(llvm::raw_ostream &os,
                llvm::StringRef functionName) const override;

  VerificationResult
  verify(const std::filesystem::path &sourceFile) const override;
};

} // namespace

std::unique_ptr<AbstractWorker>
BitwidthOptimizationsTarget::createWorker(const Options &options,
                                          Randomly randomly) const {
  return std::make_unique<BitwidthOptimizationsGenerator>(options,
                                                          std::move(randomly));
}

void BitwidthOptimizationsGenerator::generate(
    llvm::raw_ostream &os, llvm::StringRef functionName) const {
  // Enforce a strict bitwidth requirement for the entire program.
  auto maxBitwidth = random.getInteger<std::uint8_t>(1, 32);
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

AbstractWorker::VerificationResult BitwidthOptimizationsGenerator::verify(
    const std::filesystem::path &sourceFile) const {
  return performDifferentialTesting(sourceFile, options.dynamaticPath);
}
