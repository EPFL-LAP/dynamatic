//===- dynamatic++.cpp - The utility for performing dynamic HLS -----------===//
//
// This file implements 'dynamatic++'.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Chrono.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "circt/Conversion/ExportVerilog.h"
#include "circt/Conversion/Passes.h"
#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/ESI/ESIPasses.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Support/LoweringOptions.h"
#include "circt/Support/LoweringOptionsParser.h"
#include "circt/Support/Version.h"
#include "circt/Transforms/Passes.h"

using namespace llvm;
using namespace mlir;
using namespace circt;

// --------------------------------------------------------------------------
// Tool options
// --------------------------------------------------------------------------

static cl::OptionCategory mainCategory("dynamatic++ Options");

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::init("-"), cl::cat(mainCategory));

static cl::opt<std::string> outputFilename(
    "o", cl::desc("Output filename, or directory for split output"),
    cl::value_desc("filename"), cl::init("-"), cl::cat(mainCategory));

static cl::opt<bool>
    allowUnregisteredDialects("allow-unregistered-dialects",
                              cl::desc("Allow unknown dialects in the input"),
                              cl::init(false), cl::Hidden,
                              cl::cat(mainCategory));

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true), cl::cat(mainCategory));

// --------------------------------------------------------------------------
// Handshake options
// --------------------------------------------------------------------------

static cl::opt<std::string>
    bufferingStrategy("buffering-strategy",
                      cl::desc("Strategy to apply. Possible values are: "
                               "cycles, allFIFO, all (default)"),
                      cl::init("all"), cl::cat(mainCategory));

static cl::opt<unsigned> bufferSize("buffer-size",
                                    cl::desc("Number of slots in each buffer"),
                                    cl::init(2), cl::cat(mainCategory));

// --------------------------------------------------------------------------
// Tool driver code
// --------------------------------------------------------------------------

/// Create a simple canonicalizer pass.
static std::unique_ptr<Pass> createSimpleCanonicalizerPass() {
  mlir::GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.enableRegionSimplification = false;
  return mlir::createCanonicalizerPass(config);
}

static LogicalResult
runFlow(PassManager &pm, ModuleOp module,
        std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {

  // Software lowering
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createConvertSCFToCFPass());

  // Memref legalization.
  pm.addPass(circt::createFlattenMemRefPass());
  pm.nest<func::FuncOp>().addPass(
      circt::handshake::createHandshakeLegalizeMemrefsPass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.nest<handshake::FuncOp>().addPass(createSimpleCanonicalizerPass());

  // DHLS conversion
  pm.addPass(circt::createStandardToHandshakePass(false, false));
  pm.addPass(circt::handshake::createHandshakeLowerExtmemToHWPass(false));

  // Handshake transformations
  pm.nest<handshake::FuncOp>().addPass(createSimpleCanonicalizerPass());
  pm.nest<handshake::FuncOp>().addPass(
      handshake::createHandshakeMaterializeForksSinksPass());
  pm.nest<handshake::FuncOp>().addPass(createSimpleCanonicalizerPass());
  pm.nest<handshake::FuncOp>().addPass(
      handshake::createHandshakeInsertBuffersPass(bufferingStrategy,
                                                  bufferSize));
  pm.nest<handshake::FuncOp>().addPass(createSimpleCanonicalizerPass());

  // Handshake-to-SystemVerilog lowering
  pm.nest<handshake::FuncOp>().addPass(createSimpleCanonicalizerPass());
  pm.addPass(circt::createHandshakeToHWPass());
  pm.addPass(createSimpleCanonicalizerPass());

  // ESI lowering
  pm.addPass(circt::esi::createESIPortLoweringPass());
  pm.addPass(circt::esi::createESIPhysicalLoweringPass());
  pm.addPass(circt::esi::createESItoHWPass());

  // seq lowering
  pm.addPass(createSimpleCanonicalizerPass());
  pm.nest<hw::HWModuleOp>().addPass(circt::seq::createLowerSeqHLMemPass());
  pm.nest<hw::HWModuleOp>().addPass(seq::createSeqFIRRTLLowerToSVPass());
  pm.addPass(sv::createHWMemSimImplPass(false, false));
  pm.addPass(seq::createSeqLowerToSVPass());

  // HW cleanup and legalization
  pm.nest<hw::HWModuleOp>().addPass(sv::createHWCleanupPass());
  pm.nest<hw::HWModuleOp>().addPass(sv::createHWLegalizeModulesPass());
  pm.addPass(createSimpleCanonicalizerPass());
  auto &modulePM = pm.nest<hw::HWModuleOp>();
  modulePM.addPass(sv::createPrettifyVerilogPass());

  // SystemVerilog emission
  LoweringOptionsOption loweringOptions(mainCategory);
  if (loweringOptions.getNumOccurrences())
    loweringOptions.setAsAttribute(module);
  pm.addPass(createExportVerilogPass((*outputFile)->os()));

  // Go execute!
  if (failed(pm.run(module)))
    return failure();

  return success();
}

/// Process a single buffer of the input.
static LogicalResult processBuffer(
    MLIRContext &context, TimingScope &ts, llvm::SourceMgr &sourceMgr,
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  // Parse the input.
  mlir::OwningOpRef<mlir::ModuleOp> module;
  auto parserTimer = ts.nest("MLIR Parser");
  module = parseSourceFile<ModuleOp>(sourceMgr, &context);

  if (!module)
    return failure();

  // Apply any pass manager command line options.
  PassManager pm(&context);
  pm.enableVerifier(verifyPasses);
  pm.enableTiming(ts);
  applyPassManagerCLOptions(pm);

  if (failed(runFlow(pm, module.get(), outputFile)))
    return failure();

  // We intentionally "leak" the Module into the MLIRContext instead of
  // deallocating it.  There is no need to deallocate it right before process
  // exit.
  (void)module.release();
  return success();
}

/// Process a single split of the input. This allocates a source manager and
/// creates a regular or verifying diagnostic handler, depending on whether
/// the user set the verifyDiagnostics option.
static LogicalResult
processInput(MLIRContext &context, TimingScope &ts,
             std::unique_ptr<llvm::MemoryBuffer> buffer,
             std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  return processBuffer(context, ts, sourceMgr, outputFile);
}

static LogicalResult execute(MLIRContext &context) {
  if (allowUnregisteredDialects)
    context.allowUnregisteredDialects();

  // Create the timing manager we use to sample execution times.
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  auto ts = tm.getRootScope();

  // Set up the input file.
  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
  outputFile.emplace(openOutputFile(outputFilename, &errorMessage));
  if (!*outputFile) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  // Process the input.
  if (failed(processInput(context, ts, std::move(input), outputFile)))
    return failure();

  // If the result succeeded and we're emitting a file, close it.
  if (outputFile.has_value())
    (*outputFile)->keep();

  return success();
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  setBugReportMsg("");

  // Hide default LLVM options, other than for this tool.
  // MLIR options are added below.
  cl::HideUnrelatedOptions(mainCategory);

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "CIRCT HLS tool\n");

  DialectRegistry registry;
  // Register MLIR dialects.
  registry.insert<mlir::AffineDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::scf::SCFDialect>();

  // Register MLIR passes.
  mlir::registerCSEPass();
  mlir::registerSCCPPass();
  mlir::registerInlinerPass();
  mlir::registerCanonicalizerPass();

  // Register CIRCT dialects.
  registry.insert<firrtl::FIRRTLDialect, hw::HWDialect, comb::CombDialect,
                  seq::SeqDialect, sv::SVDialect, handshake::HandshakeDialect,
                  esi::ESIDialect>();

  // Do the guts of the hlstool process.
  MLIRContext context(registry);
  auto result = execute(context);

  // Use "exit" instead of return'ing to signal completion.  This avoids
  // invoking the MLIRContext destructor, which spends a bunch of time
  // deallocating memory etc which process exit will do for us.
  exit(failed(result));
}
