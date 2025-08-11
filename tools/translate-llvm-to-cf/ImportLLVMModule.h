#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringRef.h"
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

  /// LLVM -> MLIR basic block mapping.
  mlir::DenseMap<llvm::BasicBlock *, mlir::Block *> blockMapping;

  /// Mapping LLVM instruction values to MLIR results.
  mlir::DenseMap<llvm::Value *, mlir::Value> valueMapping;

  /// In MLIR memref, globals are identified using the sym_name.
  mlir::DenseMap<llvm::Value *, memref::GlobalOp> globalValToGlobalOpMap;

  /// The (C-code-level) argument types of the LLVM functions.
  FuncNameToCFuncArgsMap &argMap;

  void addMapping(llvm::Value *llvmVal, mlir::Value mlirVal) {
    valueMapping[llvmVal] = mlirVal;
  }

  std::set<Instruction *> converted;

  /// Construct an op without adding any attributes. TODO: maybe return the op
  /// to enable it?
  template <typename MLIRTy>
  void naiveTranslation(mlir::Type returnType, mlir::ValueRange values,
                        Instruction *inst) {
    MLIRTy op =
        builder.create<MLIRTy>(UnknownLoc::get(ctx), returnType, values);

    // Register the corresponding MLIR value of the result of the original
    // instruction.
    valueMapping[inst] = op.getResult();
  }

  void initializeBlocksAndBlockMapping(llvm::Function *llvmFunc,
                                       func::FuncOp funcOp);

  /// LLVM embeds constants into the instructions, where in MLIR we need to
  /// explicitly create them.
  void createConstants(llvm::Function *llvmFunc);

  /// LLVM GEP instructions can directly take a global pointer. We need to
  /// explicitly create memref::GetGlobalOp here
  void createGetGlobals(llvm::Function *llvmFunc);
  void translateFunction(llvm::Function *llvmFunc);

  void translateGlobalVars();

  // Dispatches to specialized functions:
  void translateInstruction(llvm::Instruction *inst);

  // Specialized translation functions:
  void translateBinaryInst(llvm::BinaryOperator *inst);
  void translateCastInst(llvm::CastInst *inst);
  void translateICmpInst(llvm::ICmpInst *inst);
  void translateFCmpInst(llvm::FCmpInst *inst);
  void translateBranchInst(llvm::BranchInst *inst);
  void translateGEPInst(llvm::GetElementPtrInst *gepInst);
  void translateLoadWithZeroIndices(llvm::LoadInst *loadInst);
  void translateStoreWithZeroIndices(llvm::StoreInst *storeInst);
  void translateAllocaInst(llvm::AllocaInst *allocaInst);

  SmallVector<mlir::Value> getBranchOperandsForCFGEdge(BasicBlock *currentBB,
                                                       BasicBlock *nextBB);

public:
  ImportLLVMModule(llvm::Module *llvmModule, mlir::ModuleOp mlirModule,
                   OpBuilder &builder, FuncNameToCFuncArgsMap &argMap,
                   MLIRContext *ctx, StringRef funcName)
      : funcName(funcName), mlirModule(mlirModule), llvmModule(llvmModule),
        builder(builder), ctx(ctx), argMap(argMap){};

  void translateModule();
};