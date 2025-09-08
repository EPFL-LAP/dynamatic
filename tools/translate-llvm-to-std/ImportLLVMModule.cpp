#include "ImportLLVMModule.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/MemoryDependency.h"

/// Returns the corresponding scalar MLIR Type from a given LLVM type
static mlir::Type getMLIRType(llvm::Type *llvmType,
                              mlir::MLIRContext *context) {
  if (llvmType->isIntegerTy()) {
    return mlir::IntegerType::get(context, llvmType->getIntegerBitWidth());
  }
  if (llvmType->isFloatTy()) {
    return mlir::FloatType::getF32(context);
  }
  if (llvmType->isDoubleTy()) {
    return mlir::FloatType::getF64(context);
  }

  llvm_unreachable("Unhandled scalar type");
}

/// NOTE: This is taken literally from "mlir/lib/Target/LLVMIR/ModuleImport.cpp"
///
/// Get a topologically sorted list of blocks for the given function.
/// "for (auto &BB : func)" may give you an instruction before its operand has
/// been visited.
static llvm::SetVector<llvm::BasicBlock *>
getTopologicallySortedBlocks(llvm::Function *func) {
  llvm::SetVector<llvm::BasicBlock *> blocks;
  for (llvm::BasicBlock &bb : *func) {
    if (!blocks.contains(&bb)) {
      llvm::ReversePostOrderTraversal<llvm::BasicBlock *> traversal(&bb);
      blocks.insert(traversal.begin(), traversal.end());
    }
  }

  // NOTE: This condition is triggered when there are basic blocks with
  // self-loops. LLVM IR like this should be considered as malformed and
  // should be rejected.
  assert(blocks.size() == func->size() && "some blocks are not sorted");

  return blocks;
}

static void translateMemDepAndNameAttrs(llvm::Instruction *inst, Operation *op,
                                        MLIRContext &ctx, OpBuilder &builder) {
  // Dynamatic marks memory dependency as metadata nodes in LLVM IR. This
  // function convert them to the corresponding MLIR attributes in MLIR.
  // TODOs:
  // - Separate the handling for "handshake.name" and "deps"
  // - Put MemoryDependency.h to this translation tool
  if (auto depData = LLVMMemDependency::fromLLVMInstruction(inst);
      depData.has_value()) {
    std::string opName = depData.value().name;
    SmallVector<dynamatic::handshake::MemDependenceAttr> deps =
        depData.value().getMemoryDependenceAttrs(ctx);
    op->setAttr(StringRef(dynamatic::NameAnalysis::ATTR_NAME),
                builder.getStringAttr(opName));
    if (!deps.empty()) {
      dynamatic::setDialectAttr<dynamatic::handshake::MemDependenceArrayAttr>(
          op, &ctx, deps);
    }
  }
}

static void
convertInitializerToDenseElemAttrRecursive(llvm::Constant *constValue,
                                           SmallVector<mlir::Attribute> &values,
                                           const mlir::Type &baseMLIRElemType);

// Unlike LLVM, in MLIR the dense data array itself doesn't have a particular
// shape, so we need to do this flattening conversion for multidimensional
// arrays.
//
// Converts a LLVM IR multidimension array initialization constant data into an
// 1D MLIR dense array in row major. This function recursively visits the
// dimensions and push the scalar elements into a flat array (values).
static DenseElementsAttr
convertInitializerToDenseElemAttr(llvm::GlobalVariable *globVar,
                                  MLIRContext *ctx) {
  auto *baseElementType = globVar->getInitializer()->getType();
  SmallVector<int64_t> shape;
  int64_t numElems = 1;
  while (baseElementType->isArrayTy()) {
    int64_t numElemCurrDim = baseElementType->getArrayNumElements();
    shape.push_back(numElemCurrDim);
    baseElementType = baseElementType->getArrayElementType();
    numElems *= numElemCurrDim;
  }
  auto baseMLIRElemType = getMLIRType(baseElementType, ctx);
  SmallVector<mlir::Attribute> values;
  values.reserve(numElems);
  convertInitializerToDenseElemAttrRecursive(globVar->getInitializer(), values,
                                             baseMLIRElemType);
  return mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get(shape, baseMLIRElemType), values);
}

void convertInitializerToDenseElemAttrRecursive(
    llvm::Constant *constValue, SmallVector<mlir::Attribute> &values,
    const mlir::Type &baseMLIRElemType) {
  auto *arrType = llvm::dyn_cast<llvm::ArrayType>(constValue->getType());
  for (unsigned i = 0; i < arrType->getNumElements(); ++i) {
    auto *elem = constValue->getAggregateElement(i);
    if (auto *constInt = llvm::dyn_cast<llvm::ConstantInt>(elem)) {
      values.push_back(
          mlir::IntegerAttr::get(baseMLIRElemType, constInt->getSExtValue()));
    } else if (auto *constFloat = llvm::dyn_cast<llvm::ConstantFP>(elem)) {
      values.push_back(
          mlir::FloatAttr::get(baseMLIRElemType, constFloat->getValueAPF()));
    } else if (llvm::isa<llvm::ConstantDataArray>(elem)) {
      convertInitializerToDenseElemAttrRecursive(elem, values,
                                                 baseMLIRElemType);
    } else {
      llvm::errs() << "Unhandled constant element type:\n";
      llvm_unreachable("Unhandled base element type.");
    }
  }
}

void ImportLLVMModule::translateModule() {
  translateGlobalVars();

  for (auto &f : llvmModule->functions()) {
    if (f.isDeclaration())
      continue;

    if (f.getName() != funcName)
      continue;

    translateFunction(&f);
  }
}

void ImportLLVMModule::translateFunction(llvm::Function *llvmFunc) {

  SmallVector<mlir::Type> argTypes =
      getFuncArgTypes(llvmFunc->getName().str(), argMap, builder);

  builder.setInsertionPointToEnd(mlirModule.getBody());

  SmallVector<mlir::Type> resTypes;

  if (!llvmFunc->getReturnType()->isVoidTy()) {
    resTypes.push_back(getMLIRType(llvmFunc->getReturnType(), ctx));
  }

  auto funcType = builder.getFunctionType(argTypes, resTypes);
  auto funcOp = builder.create<func::FuncOp>(builder.getUnknownLoc(),
                                             llvmFunc->getName(), funcType);

  initializeBlocksAndBlockMapping(llvmFunc, funcOp);

  auto &entryBlock = funcOp.getBody().front();
  builder.setInsertionPointToStart(&entryBlock);

  createConstants(llvmFunc);
  createGetGlobals(llvmFunc);

  llvm::SetVector<llvm::BasicBlock *> blocks =
      getTopologicallySortedBlocks(llvmFunc);
  for (auto *block : blocks) {
    builder.setInsertionPointToEnd(blockMap[block]);
    for (auto &inst : *block) {
      translateInstruction(&inst);
    }
  }
}

void ImportLLVMModule::translateGlobalVars() {
  builder.setInsertionPointToEnd(mlirModule.getBody());
  for (auto &constant : llvmModule->global_values()) {

    auto *globalVar = dyn_cast<llvm::GlobalVariable>(&constant);

    if (!globalVar)
      continue;

    auto *baseElemType = globalVar->getValueType();
    SmallVector<int64_t> shape;
    while (baseElemType->isArrayTy()) {
      shape.push_back(baseElemType->getArrayNumElements());
      baseElemType = baseElemType->getArrayElementType();
    }
    auto baseMLIRElemType = getMLIRType(baseElemType, ctx);
    auto memrefType = MemRefType::get(shape, baseMLIRElemType);

    StringRef symName = constant.getName();
    StringAttr symNameAttr = StringAttr::get(ctx, Twine(symName));
    // TODO: handling public visibility? Maybe we don't care about this?
    StringAttr visibilityAttr = StringAttr::get(ctx, Twine("private"));
    TypeAttr typeAttr = TypeAttr::get(memrefType);
    UnitAttr isConstantAttr = UnitAttr::get(ctx);
    IntegerAttr alignmentAttr =
        builder.getI64IntegerAttr(globalVar->getAlignment());
    DenseElementsAttr initialValueAttr;

    if (globalVar->hasInitializer()) {
      initialValueAttr = convertInitializerToDenseElemAttr(globalVar, ctx);
    }

    auto globalOp = builder.create<memref::GlobalOp>(
        // clang-format off
        UnknownLoc::get(ctx),
        symNameAttr,
        visibilityAttr,
        typeAttr,
        initialValueAttr,
        isConstantAttr,
        alignmentAttr
        // clang-format on
    );
    globalValToGlobalOpMap[&constant] = globalOp;
  }
}

void ImportLLVMModule::translateInstruction(llvm::Instruction *inst) {
  Location loc = UnknownLoc::get(ctx);
  if (auto *binaryOp = dyn_cast<llvm::BinaryOperator>(inst)) {
    translateBinaryInst(binaryOp);
  } else if (auto *castOp = dyn_cast<llvm::CastInst>(inst)) {
    translateCastInst(castOp);
  } else if (auto *gepInst = dyn_cast<llvm::GetElementPtrInst>(inst)) {
    translateGEPInst(gepInst);
  } else if (auto *loadInst = dyn_cast<llvm::LoadInst>(inst)) {
    translateLoadInst(loadInst);
  } else if (auto *storeInst = dyn_cast<llvm::StoreInst>(inst)) {
    translateStoreInst(storeInst);
  } else if (auto *icmpInst = dyn_cast<llvm::ICmpInst>(inst)) {
    translateICmpInst(icmpInst);
  } else if (auto *fcmpInst = dyn_cast<FCmpInst>(inst)) {
    translateFCmpInst(fcmpInst);
  } else if (auto *selInst = dyn_cast<llvm::SelectInst>(inst)) {
    mlir::Value condition = valueMap[selInst->getOperand(0)];
    mlir::Value trueOperand = valueMap[selInst->getOperand(1)];
    mlir::Value falseOperand = valueMap[selInst->getOperand(2)];
    mlir::Type resType = getMLIRType(selInst->getType(), ctx);
    naiveTranslation<arith::SelectOp>(
        // clang-format off
        resType,
        {condition, trueOperand, falseOperand},
        selInst
        // clang-format on
    );
  } else if (auto *branchInst = dyn_cast<llvm::BranchInst>(inst)) {
    translateBranchInst(branchInst);
  } else if (auto *allocaOp = dyn_cast<llvm::AllocaInst>(inst)) {
    translateAllocaInst(allocaOp);
  } else if (auto *returnOp = dyn_cast<llvm::ReturnInst>(inst)) {
    if (returnOp->getNumOperands() == 1) {
      mlir::Value arg = valueMap[inst->getOperand(0)];
      builder.create<func::ReturnOp>(loc, arg);
    } else {
      builder.create<func::ReturnOp>(loc);
    }
  } else if (isa<llvm::PHINode>(inst)) {
    // At this stage, Phi nodes are all converted to the block arguments
    return;
  } else if (auto *callInst = dyn_cast<llvm::CallInst>(inst)) {
    translateCallInst(callInst);
  } else {
    llvm_unreachable("Not implemented");
  }
}

SmallVector<mlir::Value>
ImportLLVMModule::getBranchOperandsForCFGEdge(BasicBlock *currBB,
                                              BasicBlock *nextBB) {
  SmallVector<mlir::Value> operands;
  for (PHINode &phi : nextBB->phis()) {
    mlir::Value argument = valueMap[phi.getIncomingValueForBlock(currBB)];
    operands.push_back(argument);
  }
  return operands;
}

void ImportLLVMModule::initializeBlocksAndBlockMapping(llvm::Function *llvmFunc,
                                                       func::FuncOp funcOp) {
  // Convert the entry block (specially handled, because its arguments are also
  // the function arguments). NOTE: "funcOp.addEntryBlock()" automatically adds
  // the block arguments of the first block according to the predefined function
  // signiture.
  Block *entryBlock = funcOp.addEntryBlock();
  BasicBlock &entryBB = llvmFunc->getEntryBlock();
  blockMap[&entryBB] = entryBlock;

  if (llvmFunc->arg_size() != entryBlock->getNumArguments()) {
    llvm_unreachable(
        "Inferred fewer arguments for MLIR function! The original "
        "C function potentially contains unhandled argument types.");
  }

  for (auto [llvmArg, mlirArg] :
       llvm::zip_equal(llvmFunc->args(), entryBlock->getArguments())) {
    valueMap[&llvmArg] = mlirArg;
  }

  // Remaining basic blocks
  for (BasicBlock &bb : drop_begin(*llvmFunc)) {
    auto *block = funcOp.addBlock();
    assert(!blockMap.count(&bb));
    blockMap[&bb] = block;
    for (auto &phi : bb.phis()) {
      auto argType = getMLIRType(phi.getType(), ctx);
      auto mlirArg = block->addArgument(argType, mlir::UnknownLoc::get(ctx));
      valueMap[&phi] = mlirArg;
    }
  }
}

void ImportLLVMModule::createConstants(llvm::Function *llvmFunc) {
  for (auto &block : *llvmFunc) {
    for (auto &inst : block) {
      for (unsigned i = 0; i < inst.getNumOperands(); ++i) {

        Location loc = UnknownLoc::get(ctx);
        llvm::Value *val = inst.getOperand(i);

        // NOTE: llvm use the same pointer for multiple uses of the same
        // constant, therefore we don't need to add it twice.
        if (!isa<llvm::Constant>(val) || valueMap.count(val)) {
          continue;
        }

        if (auto *intConst = dyn_cast<ConstantInt>(val)) {
          APInt intVal = intConst->getValue();
          auto constOp = builder.create<arith::ConstantIntOp>(
              loc, intVal.getSExtValue(), intVal.getBitWidth());
          valueMap[val] = constOp->getResult(0);
          loc = constOp->getLoc();
        }

        if (auto *floatConst = dyn_cast<llvm::ConstantFP>(val)) {
          const APFloat &floatVal = floatConst->getValue();
          if (&floatVal.getSemantics() == &llvm::APFloat::IEEEsingle()) {
            auto constOp = builder.create<arith::ConstantFloatOp>(
                loc, floatVal, builder.getF32Type());
            valueMap[val] = constOp->getResult(0);
            loc = constOp->getLoc();
          } else if (&floatVal.getSemantics() == &llvm::APFloat::IEEEdouble()) {
            auto constOp = builder.create<arith::ConstantFloatOp>(
                loc, floatVal, builder.getF64Type());
            valueMap[val] = constOp->getResult(0);
            loc = constOp->getLoc();
          }
        }
      }
    }
  }
}

void ImportLLVMModule::createGetGlobals(llvm::Function *llvmFunc) {
  for (auto &block : *llvmFunc) {
    for (auto &inst : block) {
      for (unsigned i = 0; i < inst.getNumOperands(); ++i) {

        Location loc = UnknownLoc::get(ctx);
        llvm::Value *val = inst.getOperand(i);

        if (auto *globalVar = dyn_cast<llvm::GlobalVariable>(val)) {

          auto globalOp = globalValToGlobalOpMap[globalVar];

          auto memrefType = globalOp.getType();

          auto getGlobalOp = builder.create<memref::GetGlobalOp>(
              loc, memrefType, globalOp.getSymName());

          valueMap[val] = getGlobalOp.getResult();
        }
      }
    }
  }
}

void ImportLLVMModule::translateBinaryInst(llvm::BinaryOperator *inst) {
  mlir::Value lhs = valueMap[inst->getOperand(0)];
  mlir::Value rhs = valueMap[inst->getOperand(1)];
  mlir::Type resType = getMLIRType(inst->getType(), ctx);
  switch (inst->getOpcode()) {
    // clang-format off
    case Instruction::Add:  naiveTranslation<arith::AddIOp>( resType,  {lhs, rhs}, inst); break;
    case Instruction::Sub:  naiveTranslation<arith::SubIOp>( resType,  {lhs, rhs}, inst); break;
    case Instruction::Mul:  naiveTranslation<arith::MulIOp>( resType,  {lhs, rhs}, inst); break;
    case Instruction::UDiv: naiveTranslation<arith::DivUIOp>( resType,  {lhs, rhs}, inst); break;
    case Instruction::SDiv: naiveTranslation<arith::DivSIOp>( resType,  {lhs, rhs}, inst); break;
    case Instruction::SRem: naiveTranslation<arith::RemSIOp>( resType,  {lhs, rhs}, inst); break;
    case Instruction::URem: naiveTranslation<arith::RemUIOp>( resType,  {lhs, rhs}, inst); break;
    case Instruction::FAdd: naiveTranslation<arith::AddFOp>( resType,  {lhs, rhs}, inst); break;
    case Instruction::FSub: naiveTranslation<arith::SubFOp>( resType,  {lhs, rhs}, inst); break;
    case Instruction::FDiv: naiveTranslation<arith::DivFOp>( resType,  {lhs, rhs}, inst); break;
    case Instruction::FMul: naiveTranslation<arith::MulFOp>( resType,  {lhs, rhs}, inst); break;
    case Instruction::Shl:  naiveTranslation<arith::ShLIOp>( resType,  {lhs, rhs}, inst); break;
    case Instruction::AShr: naiveTranslation<arith::ShRSIOp>( resType, {lhs, rhs}, inst); break;
    case Instruction::LShr: naiveTranslation<arith::ShRUIOp>( resType, {lhs, rhs}, inst); break;
    case Instruction::And:  naiveTranslation<arith::AndIOp>( resType, {lhs, rhs}, inst); break;
    case Instruction::Or:   naiveTranslation<arith::OrIOp>( resType, {lhs, rhs}, inst); break;
    case Instruction::Xor:  naiveTranslation<arith::XOrIOp>( resType, {lhs, rhs}, inst); break;
    // clang-format on
  default: {
    llvm::errs() << "Not yet handled binary operation type "
                 << inst->getOpcodeName() << "\n";
    llvm_unreachable("Not handled operation type");
  }
  }
}

void ImportLLVMModule::translateCastInst(llvm::CastInst *inst) {
  mlir::Value arg = valueMap[inst->getOperand(0)];
  mlir::Type resType = getMLIRType(inst->getType(), ctx);

  switch (inst->getOpcode()) {
    // clang-format off
    case Instruction::ZExt:    naiveTranslation<arith::ExtUIOp>( resType, {arg}, inst); break;
    case Instruction::SExt:    naiveTranslation<arith::ExtSIOp>( resType, {arg}, inst); break;
    case Instruction::FPExt:   naiveTranslation<arith::ExtFOp>( resType, {arg}, inst); break;
    case Instruction::Trunc:   naiveTranslation<arith::TruncIOp>( resType, {arg}, inst); break;
    case Instruction::FPTrunc: naiveTranslation<arith::TruncFOp>( resType, {arg}, inst); break;
    case Instruction::SIToFP:  naiveTranslation<arith::SIToFPOp>( resType, {arg}, inst); break;
    case Instruction::FPToSI:  naiveTranslation<arith::FPToSIOp>( resType, {arg}, inst); break;
    case Instruction::FPToUI:  naiveTranslation<arith::FPToUIOp>( resType, {arg}, inst); break;
    case Instruction::UIToFP:  naiveTranslation<arith::UIToFPOp>( resType, {arg}, inst); break;
    // clang-format on
  default: {
    llvm::errs() << "Not yet handled unary operation type "
                 << inst->getOpcodeName() << "\n";
    llvm_unreachable("Not implemented");
  }
  }
}

void ImportLLVMModule::translateICmpInst(llvm::ICmpInst *inst) {
  mlir::Value lhs = valueMap[inst->getOperand(0)];
  mlir::Value rhs = valueMap[inst->getOperand(1)];
  arith::CmpIPredicate predicate;
  switch (inst->getPredicate()) {
    // clang-format off
    case llvm::CmpInst::Predicate::ICMP_EQ:  predicate = arith::CmpIPredicate::eq;  break;
    case llvm::CmpInst::Predicate::ICMP_NE:  predicate = arith::CmpIPredicate::ne;  break;
    case llvm::CmpInst::Predicate::ICMP_UGT: predicate = arith::CmpIPredicate::ugt; break;
    case llvm::CmpInst::Predicate::ICMP_UGE: predicate = arith::CmpIPredicate::uge; break;
    case llvm::CmpInst::Predicate::ICMP_ULT: predicate = arith::CmpIPredicate::ult; break;
    case llvm::CmpInst::Predicate::ICMP_ULE: predicate = arith::CmpIPredicate::ule; break;
    case llvm::CmpInst::Predicate::ICMP_SGT: predicate = arith::CmpIPredicate::sgt; break;
    case llvm::CmpInst::Predicate::ICMP_SGE: predicate = arith::CmpIPredicate::sge; break;
    case llvm::CmpInst::Predicate::ICMP_SLT: predicate = arith::CmpIPredicate::slt; break;
    case llvm::CmpInst::Predicate::ICMP_SLE: predicate = arith::CmpIPredicate::sle; break;

    default: llvm_unreachable("Unsupported ICMP predicate");
    // clang-format on
  }

  auto op =
      builder.create<arith::CmpIOp>(UnknownLoc::get(ctx), predicate, lhs, rhs);
  valueMap[inst] = op->getResult(0);
}

void ImportLLVMModule::translateFCmpInst(llvm::FCmpInst *inst) {
  mlir::Value lhs = valueMap[inst->getOperand(0)];
  mlir::Value rhs = valueMap[inst->getOperand(1)];
  arith::CmpFPredicate predicate;
  switch (inst->getPredicate()) {
    // clang-format off
    case llvm::CmpInst::FCMP_OEQ: predicate = arith::CmpFPredicate::OEQ; break;
    case llvm::CmpInst::FCMP_OGT: predicate = arith::CmpFPredicate::OGT; break;
    case llvm::CmpInst::FCMP_OGE: predicate = arith::CmpFPredicate::OGE; break;
    case llvm::CmpInst::FCMP_OLT: predicate = arith::CmpFPredicate::OLT; break;
    case llvm::CmpInst::FCMP_OLE: predicate = arith::CmpFPredicate::OLE; break;
    case llvm::CmpInst::FCMP_ONE: predicate = arith::CmpFPredicate::ONE; break;
    case llvm::CmpInst::FCMP_ORD: predicate = arith::CmpFPredicate::ORD; break;
    case llvm::CmpInst::FCMP_UNO: predicate = arith::CmpFPredicate::UNO; break;
    case llvm::CmpInst::FCMP_UEQ: predicate = arith::CmpFPredicate::UEQ; break;
    case llvm::CmpInst::FCMP_UGT: predicate = arith::CmpFPredicate::UGT; break;
    case llvm::CmpInst::FCMP_UGE: predicate = arith::CmpFPredicate::UGE; break;
    case llvm::CmpInst::FCMP_ULT: predicate = arith::CmpFPredicate::ULT; break;
    case llvm::CmpInst::FCMP_ULE: predicate = arith::CmpFPredicate::ULE; break;
    case llvm::CmpInst::FCMP_UNE: predicate = arith::CmpFPredicate::UNE; break;
    case llvm::CmpInst::FCMP_FALSE: predicate = arith::CmpFPredicate::AlwaysFalse; break;
    case llvm::CmpInst::FCMP_TRUE:  predicate = arith::CmpFPredicate::AlwaysTrue; break;

    default: llvm_unreachable("Unsupported FCMP predicate");
    // clang-format on
  }
  auto op =
      builder.create<arith::CmpFOp>(UnknownLoc::get(ctx), predicate, lhs, rhs);
  valueMap[inst] = op->getResult(0);
}

void ImportLLVMModule::translateGEPInst(llvm::GetElementPtrInst *gepInst) {
  // NOTE: this function does not create any corresponding op in the CF MLIR but
  // only computes the indices for the LOAD/STORE ops that the gepInst drives.

  // Check if the GEP is not chained
  mlir::Value baseAddress = valueMap[gepInst->getPointerOperand()];
  SmallVector<mlir::Value> indexOperands;

  auto memrefType = baseAddress.getType().dyn_cast<MemRefType>();

  if (!memrefType)
    llvm_unreachable("GEP should take memref as reference");

  for (auto &indexUse : gepInst->indices()) {
    llvm::Value *indexValue = indexUse;
    mlir::Value mlirIndexValue = valueMap[indexValue];
    // NOTE: memref::LoadOp and memref::StoreOp expect their indices to be of
    // IndexType. Therefore, we cast the i32/i64 indices to IndexType. This
    // pattern will later be folded in the bitwidth optimization pass.
    auto idxCastOp = builder.create<arith::IndexCastOp>(
        UnknownLoc::get(ctx), builder.getIndexType(), mlirIndexValue);
    indexOperands.push_back(idxCastOp);
  }

  if (auto *defInst = gepInst->getPointerOperand();
      isa_and_nonnull<AllocaInst>(defInst) ||
      isa_and_nonnull<GlobalVariable>(defInst)) {
    // NOTE: If GEP calculates value from a memory allocation (which is a
    // global value), an extra zero index value is required at the beginning
    // to calculate the address.
    //
    // Reference:
    // https://llvm.org/docs/GetElementPtr.html#why-is-the-extra-0-index-required
    //
    // Therefore, we drop the first element in this case
    indexOperands.erase(indexOperands.begin());
  }

  // NOTE: GEPOp has the following syntax (some details omitted):
  // GEPOp %basePtr, %firstDim, %secondDim, %thirdDim, ...
  // When you iterate through the indices, it also returns indices from left
  // to right. However, the following two syntaxes are equivalent in LLVM:
  // - (1) GEPop %basePtr, %firstDim, 0, 0
  // - (2) GEPop %basePtr, %firstDim
  // Notice that, in the second example, the trailing constant 0s are omitted.
  // Source:
  // https://llvm.org/docs/GetElementPtr.html#why-do-gep-x-1-0-0-and-gep-x-1-alias
  //
  // However, memref::LoadOp and memref::StoreOp must have their indices
  // match the memref. So here we need to fill in the constant zeros.
  int remainingConstZeros = memrefType.getShape().size() - indexOperands.size();

  if (remainingConstZeros < 0) {
    llvm_unreachable(
        "GEP should only omit indices, but shouldn't have more indices than "
        "the original memref type extracted from the function argument!");
  }

  for (int i = 0; i < remainingConstZeros; i++) {
    auto constZeroOp = builder.create<arith::ConstantOp>(
        UnknownLoc::get(ctx), builder.getI64IntegerAttr(0));
    auto idxCastOp = builder.create<arith::IndexCastOp>(
        UnknownLoc::get(ctx), builder.getIndexType(), constZeroOp);
    indexOperands.push_back(idxCastOp);
  }

  this->gepInstToMemRefAndIndicesMap[gepInst] =
      MemRefAndIndices(baseAddress, indexOperands);
}

void ImportLLVMModule::translateBranchInst(llvm::BranchInst *inst) {
  BasicBlock *currLLVMBB = inst->getParent();
  Location loc = UnknownLoc::get(ctx);
  if (inst->isUnconditional()) {
    BasicBlock *nextLLVMBB = dyn_cast_or_null<BasicBlock>(inst->getOperand(0));
    assert(nextLLVMBB &&
           "The unconditional branch doesn't have a BB as operand!");
    builder.create<cf::BranchOp>(
        // clang-format off
        loc,
        blockMap[nextLLVMBB],
        getBranchOperandsForCFGEdge(currLLVMBB, nextLLVMBB)
        // clang-format on
    );
  } else {
    // NOTE: operands of the branch instruction [Cond, FalseDest,] TrueDest
    // (from the C++ API).
    BasicBlock *falseDestBB = dyn_cast_or_null<BasicBlock>(inst->getOperand(1));
    assert(falseDestBB);
    BasicBlock *trueDestBB = dyn_cast_or_null<BasicBlock>(inst->getOperand(2));
    assert(trueDestBB);
    SmallVector<mlir::Value> falseOperands =
        getBranchOperandsForCFGEdge(currLLVMBB, falseDestBB);
    SmallVector<mlir::Value> trueOperands =
        getBranchOperandsForCFGEdge(currLLVMBB, trueDestBB);
    mlir::Value condition = valueMap[inst->getCondition()];
    builder.create<cf::CondBranchOp>(
        // clang-format off
        loc,
        condition,
        blockMap[trueDestBB],
        trueOperands,
        blockMap[falseDestBB],
        falseOperands
        // clang-format on
    );
  }
}

void ImportLLVMModule::translateLoadInst(llvm::LoadInst *loadInst) {
  Location loc = UnknownLoc::get(ctx);
  auto *instAddr = loadInst->getPointerOperand();
  mlir::Value memref;
  SmallVector<mlir::Value> indices;
  if (this->gepInstToMemRefAndIndicesMap.count(instAddr)) {
    // Logic: In LLVM IR, load/store operations take the pointer computed from
    // GEP ops, whereas in memref, load operations takes indices (of index
    // type). This function uses the index operand collected when processing
    // GEPs as operands of LOAD/STOREs.
    auto memrefAndIndices = this->gepInstToMemRefAndIndicesMap[instAddr];
    memref = memrefAndIndices.first;
    indices = memrefAndIndices.second;
  } else {
    if (isa<GetElementPtrInst>(instAddr))
      llvm_unreachable(
          "Converting a load but the producer hasn't been converted yet!");
    // NOTE: This condition handles a special case where a load only has
    // constant indices, e.g., tmp = mat[0][0].
    memref = this->valueMap[instAddr];
    auto memrefType = memref.getType().dyn_cast<MemRefType>();
    int constZerosToAdd = memrefType.getShape().size();
    for (int i = 0; i < constZerosToAdd; i++) {
      auto constZeroOp = this->builder.create<arith::ConstantOp>(
          loc, this->builder.getIndexAttr(0));
      indices.push_back(constZeroOp);
    }
  }
  mlir::Type resType = getMLIRType(loadInst->getType(), ctx);
  auto newOp = builder.create<memref::LoadOp>(loc, resType, memref, indices);
  valueMap[loadInst] = newOp.getResult();
  translateMemDepAndNameAttrs(loadInst, newOp, *ctx, builder);
}

void ImportLLVMModule::translateStoreInst(llvm::StoreInst *storeInst) {
  Location loc = UnknownLoc::get(ctx);
  auto *instAddr = storeInst->getPointerOperand();

  mlir::Value memref;
  SmallVector<mlir::Value> indices;

  if (this->gepInstToMemRefAndIndicesMap.count(instAddr)) {
    // Logic: In LLVM IR, load/store operations take the pointer computed from
    // GEP ops, whereas in memref, load operations takes indices (of index
    // type). This function uses the index operand collected when processing
    // GEPs as operands of LOAD/STOREs.
    auto memrefAndIndices = this->gepInstToMemRefAndIndicesMap[instAddr];
    memref = memrefAndIndices.first;
    indices = memrefAndIndices.second;
  } else {
    // NOTE: This condition handles a special case where a load only has
    // constant indices, e.g., tmp = mat[0][0].
    if (isa<GetElementPtrInst>(instAddr))
      llvm_unreachable(
          "Converting a load but the producer hasn't been converted yet!");
    memref = this->valueMap[instAddr];
    auto memrefType = memref.getType().dyn_cast<MemRefType>();

    int constZerosToAdd = memrefType.getShape().size();
    for (int i = 0; i < constZerosToAdd; i++) {
      auto constZeroOp = this->builder.create<arith::ConstantOp>(
          loc, this->builder.getIndexAttr(0));
      indices.push_back(constZeroOp);
    }
  }

  mlir::Value storeValue = valueMap[storeInst->getValueOperand()];
  auto newOp =
      builder.create<memref::StoreOp>(loc, storeValue, memref, indices);
  translateMemDepAndNameAttrs(storeInst, newOp, *ctx, builder);
}

void ImportLLVMModule::translateAllocaInst(llvm::AllocaInst *allocaInst) {
  Location loc = UnknownLoc::get(ctx);

  SmallVector<int64_t> shape;
  llvm::Type *baseElementType = allocaInst->getAllocatedType();

  while (baseElementType->isArrayTy()) {
    shape.push_back(baseElementType->getArrayNumElements());
    baseElementType = baseElementType->getArrayElementType();
  }

  auto memrefType = MemRefType::get(shape, getMLIRType(baseElementType, ctx));

  auto allocaOp = builder.create<memref::AllocaOp>(loc, memrefType);
  valueMap[allocaInst] = allocaOp->getResult(0);
}

void ImportLLVMModule::translateCallInst(llvm::CallInst *callInst) {

  Function *calledFunc = callInst->getCalledFunction();
  assert(calledFunc);

  if (!calledFunc->isIntrinsic()) {
    assert(false && "Function calls are not currently supported");
  }

  if (calledFunc->getIntrinsicID() == Intrinsic::smax) {
    mlir::Value lhs = valueMap[callInst->getArgOperand(0)];
    mlir::Value rhs = valueMap[callInst->getArgOperand(1)];
    auto retType = getMLIRType(callInst->getType(), ctx);
    naiveTranslation<arith::MaxSIOp>(retType, {lhs, rhs}, callInst);
  } else if (calledFunc->getIntrinsicID() == Intrinsic::fabs) {
    mlir::Value arg = valueMap[callInst->getArgOperand(0)];
    auto retType = getMLIRType(callInst->getType(), ctx);
    naiveTranslation<math::AbsFOp>(retType, {arg}, callInst);
  } else {
    llvm_unreachable("Not implemented llvm intrinsic function handling!");
  }
}
