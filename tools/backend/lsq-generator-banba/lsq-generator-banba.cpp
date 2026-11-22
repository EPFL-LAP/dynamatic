#include "BanbaOps.h"
#include "LsqConfig.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

static llvm::cl::OptionCategory banbaCategory("lsq-generator-banba options");

static llvm::cl::opt<std::string>
    configFile("config", llvm::cl::desc("Path to the LSQ's JSON config file"),
               llvm::cl::Required, llvm::cl::cat(banbaCategory));

int main(int argc, char **argv) {
  // Only show this program's own options in --help
  llvm::cl::HideUnrelatedOptions(banbaCategory);
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Loading a CIRCT dialect makes this program depend on CIRCT's headers and
  // libraries, which checks that Dynamatic's build finds both
  mlir::MLIRContext context;
  context.loadDialect<circt::hw::HWDialect>();
  context.loadDialect<circt::comb::CombDialect>();
  context.loadDialect<banba::BanbaDialect>();

  // Read the config file
  auto buffer = llvm::MemoryBuffer::getFile(configFile);
  if (!buffer) {
    llvm::errs() << "Cannot read " << configFile << ": "
                 << buffer.getError().message() << "\n";
    return 1;
  }

  // Parse its content as JSON, into the struct
  llvm::Expected<LsqConfig> config =
      llvm::json::parse<LsqConfig>((*buffer)->getBuffer(), "LSQ config");
  if (!config) {
    llvm::errs() << "Cannot parse " << configFile << ": "
                 << llvm::toString(config.takeError()) << "\n";
    return 1;
  }

  // Stop if a switch of the pre-existing LSQ generators
  // has a value other than the one Banba requires
  if (!hasRequiredValues(*config))
    return 1;

  // Store the LSQ's parameters in the Banba dialect's attribute
  banba::ConfigAttr configAttr = banba::ConfigAttr::get(&context, *config);

  // Create an empty module, with the config stored on it
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  module->getOperation()->setAttr(banba::configAttrName, configAttr);

  // Make a builder which creates operations inside the module
  mlir::OpBuilder builder(&context);
  builder.setInsertionPointToEnd(module->getBody());
  mlir::Location loc = builder.getUnknownLoc();

  // Create a placeholder for each operand of the ready generation
  llvm::SmallVector<mlir::Value> placeholders;
  for (mlir::Type type :
       banba::shared::ga::ReadyGenOp::getExpectedOperandTypes(configAttr))
    placeholders.push_back(banba::PlaceholderOp::create(builder, loc, type));

  // Create the ready generation, with its expected result types
  // and the placeholders as its operands
  auto readyGen = banba::shared::ga::ReadyGenOp::create(
      builder, loc,
      banba::shared::ga::ReadyGenOp::getExpectedResultTypes(configAttr),
      placeholders);

  // Construct the inside of the ready generation
  readyGen.constructRegion();

  // Create a placeholder for each operand of the count extraction
  llvm::SmallVector<mlir::Value> countExtractPlaceholders;
  for (mlir::Type type :
       banba::shared::ga::CountExtractOp::getExpectedOperandTypes(configAttr))
    countExtractPlaceholders.push_back(
        banba::PlaceholderOp::create(builder, loc, type));

  // Create the count extraction, with its expected result types
  // and the placeholders as its operands
  auto countExtract = banba::shared::ga::CountExtractOp::create(
      builder, loc,
      banba::shared::ga::CountExtractOp::getExpectedResultTypes(configAttr),
      countExtractPlaceholders);

  // Construct the inside of the count extraction
  countExtract.constructRegion();

  // Print the module
  module->print(llvm::outs());
  return 0;
}
