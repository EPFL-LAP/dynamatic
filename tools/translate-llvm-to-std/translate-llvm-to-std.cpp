#include "dynamatic/InitAllDialects.h"
#include "dynamatic/InitAllPasses.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include "InferArgTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"

#include "ImportLLVMModule.h"
#include "mlir/Support/FileUtilities.h"

using namespace llvm;
using namespace mlir;

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input .ll file>"), cl::Required);

static cl::opt<std::string> csource("csource", cl::desc("C source file name"),
                                    cl::value_desc("filename"), cl::init("-"));

static cl::opt<std::string> funcName("function-name", cl::desc("Function name"),
                                     cl::value_desc("name"), cl::init("-"));

static cl::opt<std::string> dynamaticPath("dynamatic-path",
                                          cl::desc("Dynamatic path"),
                                          cl::value_desc("path name"),
                                          cl::init("-"));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "LLVM IR to MLIR CF converter\n");

  // Load LLVM IR
  LLVMContext llvmContext;
  SMDiagnostic err;
  std::unique_ptr<Module> llvmModule =
      parseAssemblyFile(StringRef(inputFilename), err, llvmContext);
  if (!llvmModule) {
    errs() << "Failed to read LLVM IR file.\n";
    err.print(argv[0], errs());
    return 1;
  }

  // Initialize MLIR
  mlir::DialectRegistry registry;

  registry.insert<
      // clang-format off
      func::FuncDialect,
      memref::MemRefDialect,
      arith::ArithDialect,
      math::MathDialect,
      cf::ControlFlowDialect,
      dynamatic::handshake::HandshakeDialect
      // clang-format on
      >();
  MLIRContext context(registry);

  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<math::MathDialect>();
  context.getOrLoadDialect<cf::ControlFlowDialect>();
  context.getOrLoadDialect<dynamatic::handshake::HandshakeDialect>();

  OpBuilder builder(&context);

  auto module = builder.create<ModuleOp>(builder.getUnknownLoc());

  // LLVM IR's argument does not indicate high-level types such as array shapes.
  // We use the original C code to recover this information.
  FuncNameToCFuncArgsMap nameToArgTypesMap =
      inferArgTypes(csource, dynamaticPath + "/include");

  ImportLLVMModule importer(llvmModule.get(), module, builder,
                            nameToArgTypesMap, &context, funcName);
  importer.translateModule();

  if (failed(module.verify())) {
    return 1;
  }

  std::string errorMessage;
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  mlir::OpPrintingFlags printFlags;

  AsmState state(module, OpPrintingFlags());

  module->print(output->os(), state);

  output->keep();

  return 0;
}
