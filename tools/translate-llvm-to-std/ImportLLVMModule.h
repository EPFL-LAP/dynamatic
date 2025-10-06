#include "llvm/ADT/StringRef.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"

using namespace llvm;
using namespace mlir;

using MemRefAndIndices =
    std::pair<mlir::Value /* memref */, SmallVector<mlir::Value> /* Indices */>;

class ImportLLVMModule {
public:
  ImportLLVMModule(llvm::Module *llvmModule, mlir::ModuleOp mlirModule,
                   OpBuilder &builder, FuncNameToCFuncArgsMap &argMap,
                   MLIRContext *ctx, StringRef funcName)
      : funcName(funcName), mlirModule(mlirModule), llvmModule(llvmModule),
        builder(builder), ctx(ctx), argMap(argMap) {};

  /// Calling this method will translate the funcName to mlirModule. TODO: maybe
  /// we could return a newly created "OwnOpReference<ModuleOp>" instead?
  void translateModule();

private:
  /// FIXME: This module importer only converts for one given function (name).
  StringRef funcName;

  mlir::ModuleOp mlirModule;

  llvm::Module *llvmModule;

  OpBuilder &builder;

  MLIRContext *ctx;

  /// LLVM -> MLIR basic block mapping.
  mlir::DenseMap<llvm::BasicBlock *, mlir::Block *> blockMap;

  /// In MLIR memref, globals are identified using the sym_name; in LLVM IR,
  /// globals produce values and can be read by GEPs. We use this data structure
  /// to remember the mapping.
  mlir::DenseMap<llvm::Value *, memref::GlobalOp> globalValToGlobalOpMap;

  /// Mapping LLVM instruction values to MLIR results. The values of LLVM IR to
  /// Std dialect generally have one-to-one correspondence (except for the
  /// inputs of load/store, see below).
  mlir::DenseMap<llvm::Value *, mlir::Value> valueMap;

  /// In LLVM IR to CF, we convert GEP -> LOAD/STORE to LOAD/STORE.
  /// - In LLVM IR: load and store take pointer operand
  /// - In MLIR IR: load and store take base address and indices
  /// We use this data structure to store the base address and indices provided
  /// by GEPs when processing GEPs. This will be used in LOAD/STORE conversions
  /// to lookup the input base address and indices.
  mlir::DenseMap<llvm::Value *, MemRefAndIndices> gepInstToMemRefAndIndicesMap;

  /// The (C-code-level) argument types of the LLVM functions.
  FuncNameToCFuncArgsMap &argMap;

  /// Construct an op without adding any attributes. TODO: maybe return the op
  /// to enable it?
  template <typename MLIRTy>
  void naiveTranslation(mlir::Type returnType, mlir::ValueRange values,
                        Instruction *inst) {
    MLIRTy op =
        builder.create<MLIRTy>(UnknownLoc::get(ctx), returnType, values);
    // Register the corresponding MLIR value of the result of the original
    // instruction.
    valueMap[inst] = op.getResult();
  }

  void initializeBlocksAndBlockMapping(llvm::Function *llvmFunc,
                                       func::FuncOp funcOp);

  void translateFunction(llvm::Function *llvmFunc);

  /// LLVM embeds constants into the instructions, wheres in MLIR we need to
  /// explicitly create them.
  void createConstants(llvm::Function *llvmFunc);

  /// LLVM "GlobalVariables" are converted to "memref.global"
  void translateGlobalVars();

  /// LLVM GEP instructions can directly take a global pointer. We need to
  /// explicitly create memref::GetGlobalOp here
  void createGetGlobals(llvm::Function *llvmFunc);

  /// Dispatches to specialized functions:
  void translateInstruction(llvm::Instruction *inst);

  /// Specialized translation functions:
  void translateBinaryInst(llvm::BinaryOperator *inst);
  void translateCastInst(llvm::CastInst *inst);
  void translateICmpInst(llvm::ICmpInst *inst);
  void translateFCmpInst(llvm::FCmpInst *inst);
  void translateBranchInst(llvm::BranchInst *inst);
  void translateGEPInst(llvm::GetElementPtrInst *gepInst);
  void translateLoadInst(llvm::LoadInst *loadInst);
  void translateStoreInst(llvm::StoreInst *storeInst);
  void translateAllocaInst(llvm::AllocaInst *allocaInst);
  void translateCallInst(llvm::CallInst *callInst);

  SmallVector<mlir::Value> getBranchOperandsForCFGEdge(BasicBlock *currBB,
                                                       BasicBlock *nextBB);
};
