#include "BanbaOps.h"
#include "LowerToHw.h"
#include "LsqConfig.h"
#include "circt/Conversion/ExportVerilog.h"
#include "circt/Conversion/SeqToSV.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

static llvm::cl::OptionCategory banbaCategory("lsq-generator-banba options");

static llvm::cl::opt<std::string>
    configFile("config", llvm::cl::desc("Path to the LSQ's JSON config file"),
               llvm::cl::Required, llvm::cl::cat(banbaCategory));

static llvm::cl::opt<std::string> svDir(
    "sv-dir",
    llvm::cl::desc("Directory to write the generated SystemVerilog files to"),
    llvm::cl::Required, llvm::cl::cat(banbaCategory));

/// Creates an operation with a placeholder for each operand it expects,
/// and constructs its region.
/// Temporary, until this program creates the top-level `lsq` operation
template <typename OpT>
static void createWithPlaceholders(mlir::OpBuilder &builder, mlir::Location loc,
                                   banba::ConfigAttr config) {
  // Create a placeholder for each operand of the operation
  llvm::SmallVector<mlir::Value> placeholders;
  for (mlir::Type type : OpT::getExpectedOperandTypes(config))
    placeholders.push_back(banba::PlaceholderOp::create(builder, loc, type));

  // Create the operation, with its expected result types
  // and the placeholders as its operands
  OpT op = OpT::create(builder, loc, OpT::getExpectedResultTypes(config),
                       placeholders);

  // Construct the inside of the operation
  op.constructRegion();
}

int main(int argc, char **argv) {
  // Only show this program's own options in --help
  llvm::cl::HideUnrelatedOptions(banbaCategory);
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Loading a CIRCT dialect makes this program depend on CIRCT's headers and
  // libraries, which checks that Dynamatic's build finds both
  mlir::MLIRContext context;
  context.loadDialect<circt::hw::HWDialect>();
  context.loadDialect<circt::comb::CombDialect>();
  context.loadDialect<circt::seq::SeqDialect>();
  context.loadDialect<circt::sv::SVDialect>();
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

  // Create and construct the shared group allocator
  createWithPlaceholders<banba::shared::GroupAllocatorOp>(builder, loc,
                                                          configAttr);

  // Lower the banba dialect to hw modules and instances
  banba::lowerToHw(*module);

  // Print the module
  module->print(llvm::outs());

  // Lower the seq dialect to sv before the export,
  // which turns the clock type into a plain bit.
  // CIRCT provides this lowering as an MLIR pass,
  // which can only be run through a pass manager
  mlir::PassManager seqPassManager(&context);
  seqPassManager.addPass(circt::createLowerSeqToSVPass());

  // Run the pass on the module, and stop if it fails
  if (mlir::failed(seqPassManager.run(module->getOperation()))) {
    llvm::errs() << "seq-to-sv lowering failed\n";
    return 1;
  }

  // Write the hw modules as SystemVerilog files in the given directory
  if (mlir::failed(circt::exportSplitVerilog(*module, svDir))) {
    llvm::errs() << "SystemVerilog export failed\n";
    return 1;
  }
  return 0;
}
