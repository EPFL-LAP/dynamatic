#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
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

#include "InferArgTypes.h"
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

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <vector>

using namespace llvm;
using namespace mlir;

class ImportLLVMModule {

  StringRef funcName;

  mlir::ModuleOp mlirModule;

  llvm::Module *llvmModule;

  OpBuilder &builder;

  MLIRContext *ctx;

  mlir::DenseMap<llvm::BasicBlock *, mlir::Block *> blockMapping;

  mlir::DenseMap<llvm::Value *, mlir::Value> valueMapping;

  FuncNameToCFuncArgsMap &argMap;

  void addMapping(llvm::Value *llvmVal, mlir::Value mlirVal) {
    valueMapping[llvmVal] = mlirVal;
  }

  std::set<Instruction *> converted;

  template <typename MLIRTy>
  void naiveTranslation(Location &loc, mlir::Type returnType,
                        mlir::ValueRange values, Instruction *inst) {
    MLIRTy op = builder.create<MLIRTy>(loc, returnType, values);
    addMapping(inst, op.getResult());
    loc = op.getLoc();
  }

  void initializeBlocksAndBlockMapping(llvm::Function *llvmFunc,
                                       func::FuncOp funcOp);

  void createConstants(llvm::Function *llvmFunc);
  void translateLLVMFunction(llvm::Function *llvmFunc);

  void translateOperation(llvm::Instruction *inst);

  void translateGEPOp(llvm::GetElementPtrInst *gepInst);

  SmallVector<mlir::Value> getBranchOperandsForCFGEdge(BasicBlock *currentBB,
                                                       BasicBlock *nextBB);

public:
  ImportLLVMModule(llvm::Module *llvmModule, mlir::ModuleOp mlirModule,
                   OpBuilder &builder, FuncNameToCFuncArgsMap &argMap,
                   MLIRContext *ctx, StringRef funcName)
      : funcName(funcName), mlirModule(mlirModule), llvmModule(llvmModule),
        builder(builder), ctx(ctx), argMap(argMap){};

  void translateModule() {
    for (auto &f : llvmModule->functions()) {
      if (f.isDeclaration())
        continue;

      if (f.getName() != funcName)
        continue;

      translateLLVMFunction(&f);
    }
  }
};