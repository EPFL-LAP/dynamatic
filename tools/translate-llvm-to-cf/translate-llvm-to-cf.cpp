#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
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
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Parser/Parser.h"

#include <fstream>
#include <iostream>
#include <map>
#include <vector>

using namespace llvm;
using namespace mlir;

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input .ll file>"), cl::Required);

mlir::Type convertLLVMTypeToMLIR(llvm::Type *llvmType,
                                 mlir::MLIRContext *context) {
  mlir::Type mlirType;

  if (llvmType->isIntegerTy(1)) {
    mlirType = mlir::IntegerType::get(context, 1);
  } else if (llvmType->isIntegerTy(32)) {
    mlirType = mlir::IntegerType::get(context, 32);
  } else if (llvmType->isFloatTy()) {
    mlirType = mlir::FloatType::getF32(context);
  } else if (llvmType->isDoubleTy()) {
    mlirType = mlir::FloatType::getF64(context);
  } else {
    assert(false);
  }
  return mlirType;
}

class ImportLLVMModule {
  mlir::ModuleOp mlirModule;

  llvm::Module *llvmModule;

  OpBuilder &builder;

  MLIRContext *ctx;

  mlir::DenseMap<llvm::BasicBlock *, mlir::Block *> blockMapping;

  mlir::DenseMap<llvm::Value *, mlir::Value> valueMapping;

  void addMapping(llvm::Value *llvmVal, mlir::Value mlirVal) {
    valueMapping[llvmVal] = mlirVal;
  }

public:
  ImportLLVMModule(llvm::Module *llvmModule, mlir::ModuleOp mlirModule,
                   OpBuilder &builder)
      : mlirModule(mlirModule), llvmModule(llvmModule), builder(builder) {
    ctx = mlirModule->getContext();
  };
  void translateLLVMFunction(llvm::Function *llvmFunc);
  void translateModule() {
    for (auto &f : llvmModule->functions()) {
      if (f.isDeclaration())
        continue;
      translateLLVMFunction(&f);
    }
  }

  template <typename LLVMOpType, typename MLIROpType>
  void translateOperation(LLVMOpType inst) {
    SmallVector<mlir::Value> mlirValues;
    mlir::Type resultType = convertLLVMTypeToMLIR(inst, ctx);
    for (unsigned i = 0; i < inst->getNumOperands(); ++i) {
      llvm::Value *val = inst->getOperand(i);
      mlirValues.push_back(valueMapping[val]);
    }
    builder.create<MLIROpType>(resultType, mlirValues);
    addMapping(inst, resultType);
  }
};

void ImportLLVMModule::translateLLVMFunction(llvm::Function *llvmFunc) {
  builder.setInsertionPointToEnd(mlirModule->getBlock());
  SmallVector<mlir::Type> argTypes;
  for (auto &arg : llvmFunc->args()) {
    argTypes.push_back(builder.getI32Type()); // Simplification
  }

  auto funcType = builder.getFunctionType(argTypes, {builder.getI32Type()});
  auto funcOp = builder.create<func::FuncOp>(mlirModule->getLoc(),
                                             llvmFunc->getName(), funcType);
  auto &entryBlock = *funcOp.addEntryBlock();

  builder.setInsertionPointToStart(&entryBlock);

  // Fake return 0
  auto zero =
      builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), 0, 32);
  builder.create<func::ReturnOp>(builder.getUnknownLoc(), zero.getResult());

  mlirModule.push_back(funcOp);
}

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
