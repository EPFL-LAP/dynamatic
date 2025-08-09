#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Parser/Parser.h"

#include "ImportLLVMModule.h"

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <vector>

using namespace llvm;
using namespace mlir;

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input .ll file>"), cl::Required);

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
  MLIRContext context;
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<cf::ControlFlowDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();

  OpBuilder builder(&context);
  auto module = ModuleOp::create(builder.getUnknownLoc());

  ImportLLVMModule importer(llvmModule.get(), module, builder);
  importer.translateModule();

  module.print(llvm::outs());
  return 0;
}
