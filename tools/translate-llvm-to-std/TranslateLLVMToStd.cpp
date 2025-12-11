#include "TranslateLLVMToStd.h"
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

#define DEBUG_TYPE "translate-llvm-to-std"

/// Returns the corresponding scalar MLIR Type from a given LLVM type
static mlir::Type getMLIRType(llvm::Type *llvmType,
                              mlir::MLIRContext *context) {
  if (llvmType->isIntegerTy()) {
    return mlir::IntegerType::get(context, llvmType->getIntegerBitWidth());
  }
  if (llvmType->isFloatTy()) {
    return mlir::Float32Type::get(context);
  }
  if (llvmType->isDoubleTy()) {
    return mlir::Float64Type::get(context);
  }
  // long double is mapped to F80 in LLVM (we don't have floating point units to
  // handle that).
  if (llvmType->isX86_FP80Ty()) {
    llvm::errs() << "Warning: using x86_fp80 type in MLIR translation. This "
                    "type is not currently supported\n";
    return mlir::FloatType::getF80(context);
  }
  LLVM_DEBUG(llvm::errs() << "Unhandled LLVM scalar type:\n";
             llvmType->dump(););

  llvm::report_fatal_error("Unhandled scalar type");
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
      mlir::RankedTensorType::get(/*shape = */ {numElems}, baseMLIRElemType),
      values);
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
      llvm::report_fatal_error("Unhandled base element type.");
    }
  }
}

void TranslateLLVMToStd::translateLLVMModule() {
  translateGlobalVars();

  for (auto &f : llvmModule->functions()) {
    if (f.isDeclaration())
      continue;

    if (f.getName() != funcName)
      continue;

    translateFunction(&f);
  }
}

void TranslateLLVMToStd::translateFunction(llvm::Function *llvmFunc) {

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

void TranslateLLVMToStd::translateGlobalVars() {
  builder.setInsertionPointToEnd(mlirModule.getBody());
  for (auto &constant : llvmModule->global_values()) {

    auto *globalVar = dyn_cast<llvm::GlobalVariable>(&constant);

    if (!globalVar)
      continue;

    auto *baseElemType = globalVar->getValueType();
    int64_t numElements = 1;
    while (baseElemType->isArrayTy()) {
      numElements *= baseElemType->getArrayNumElements();
      baseElemType = baseElemType->getArrayElementType();
    }
    auto baseMLIRElemType = getMLIRType(baseElemType, ctx);
    auto memrefType =
        MemRefType::get(/* shape = */ {numElements}, baseMLIRElemType);

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

void TranslateLLVMToStd::translateInstruction(llvm::Instruction *inst) {
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
TranslateLLVMToStd::getBranchOperandsForCFGEdge(BasicBlock *currBB,
                                                BasicBlock *nextBB) {
  SmallVector<mlir::Value> operands;
  for (PHINode &phi : nextBB->phis()) {
    mlir::Value argument = valueMap[phi.getIncomingValueForBlock(currBB)];
    operands.push_back(argument);
  }
  return operands;
}

void TranslateLLVMToStd::initializeBlocksAndBlockMapping(
    llvm::Function *llvmFunc, func::FuncOp funcOp) {
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

void TranslateLLVMToStd::createConstants(llvm::Function *llvmFunc) {
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

void TranslateLLVMToStd::createGetGlobals(llvm::Function *llvmFunc) {
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

void TranslateLLVMToStd::translateBinaryInst(llvm::BinaryOperator *inst) {
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

void TranslateLLVMToStd::translateCastInst(llvm::CastInst *inst) {
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

void TranslateLLVMToStd::translateICmpInst(llvm::ICmpInst *inst) {
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

void TranslateLLVMToStd::translateFCmpInst(llvm::FCmpInst *inst) {
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

void TranslateLLVMToStd::translateGEPInst(llvm::GetElementPtrInst *gepInst) {

  // The GEP instruction calculates the index that the load/store need to use
  // the access memory.
  //
  // For instance A[i][j][k] would be GEP -> ... -> GEP -> load
  //
  // In TranslateLLVMToStd, we convert it to additions and multiplications.
  //
  // Also, we flatten the memory into 1D. Since it is very hard to reliably
  // reverse engineer the order between the accesses.
  //
  // An example of flattening a multi-dimension array in a row-major order.
  // Given an array: my_array[A][B][C][D];
  // we access it as my_array[i][j][k][l];
  // The flattened index is computed as:
  //
  // (B * C * D) * i + (C * D) * j + (D) * k + l

  // Convert the GEP instruction into a series of "idx * dim + idx * dim ..."
  llvm::Type *baseElementType = gepInst->getSourceElementType();

  // Get the dimensions of the original array from the function type.
  // For the example above, it would be {A, B, C, D}
  SmallVector<int64_t> multipliers;
  while (baseElementType->isArrayTy()) {
    multipliers.push_back(baseElementType->getArrayNumElements());
    baseElementType = baseElementType->getArrayElementType();
  }

  SmallVector<llvm::Value *> gepIndices(gepInst->indices());

  mlir::Value baseAddress;
  if (this->getInstToMemRefMap.count(gepInst->getPointerOperand())) {
    // When the GEP directly gets calculates from a AllocaInst (i.e., an
    // internal array) or a pointer in the function argument (i.e.,
    // my_array[A][B][C][D] in the example above). Here we get the corresponding
    // memref in MLIR using getInstToMemRefMap.
    baseAddress = this->getInstToMemRefMap[gepInst->getPointerOperand()];
  } else {
    // Otherwise, there should be a chain of GEPs (TODO: assert this
    // assumption). The base address is the result of the previous GEP.
    baseAddress = valueMap[gepInst->getPointerOperand()];
  }
  this->getInstToMemRefMap[gepInst] = baseAddress;

  // A list of value to be accumulated. For the example above:
  // multipliedIndices = { (B * C * D) * i, (C * D) * j, (D) * k + l }
  SmallVector<mlir::Value> multipliedIndices;

  // [START calculate the flattened array indices]
  for (size_t i = 0; i < gepIndices.size(); ++i) {
    mlir::Value mlirIndexValue = valueMap[gepIndices[i]];

    // Calculate the partitial index
    int64_t coeff = 1;
    for (size_t j = i; j < multipliers.size(); j++)
      coeff *= multipliers[j];

    if (coeff == 1) {
      if (auto *constVal = dyn_cast<Constant>(gepIndices[i])) {
        // Special case: when the GEP index is a constant number
        //
        // Here we need to handle tricky examples like this:
        // %arrayidx2 = getelementptr inbounds nuw i8, ptr %2, i64 4
        //
        // Here we are using GEP to advance 4 * i8 = 32 bits. If the
        // element of the original array was 32-bit wide, then here we need to
        // increment 1 step (instead of 4).
        auto memrefType = dyn_cast<MemRefType>(baseAddress.getType());
        assert(memrefType);

        // This is the size of the actual element (i.e., for 32 in the example
        // above).
        unsigned actualBaseElementWidth = memrefType.getElementTypeBitWidth();

        // This is the size that the current GEP assumes (i.e., for i8 in the
        // example above).
        unsigned currBaseElementBitWidth =
            baseElementType->getScalarSizeInBits();

        // This the the number of `currBaseElementBitWidth` that we need to skip
        // (i.e., 4 in the example above).
        int64_t constInt = *constVal->getUniqueInteger().getRawData();

        assert((currBaseElementBitWidth * constInt) % actualBaseElementWidth ==
                   0 &&
               "Incorrect alignment!");

        unsigned actualAdvanceValue =
            (currBaseElementBitWidth * constInt) / actualBaseElementWidth;

        auto byteAlignedConstantValue = builder.create<arith::ConstantOp>(
            UnknownLoc::get(ctx),
            builder.getIntegerAttr(builder.getI64Type(), actualAdvanceValue));
        multipliedIndices.push_back(byteAlignedConstantValue);
      } else {
        multipliedIndices.push_back(mlirIndexValue);
      }

    } else if (llvm::isPowerOf2_64(coeff)) {
      // Special case: the array dimension is a power of two.
      // Here we can apply the optimization: multiply by power of 2 is the same
      // as shifting
      auto shiftValue = builder.create<arith::ConstantOp>(
          UnknownLoc::get(ctx),
          builder.getIntegerAttr(builder.getI64Type(), llvm::Log2_64(coeff)));
      auto idx = builder.create<arith::ShLIOp>(UnknownLoc::get(ctx),
                                               mlirIndexValue, shiftValue);
      multipliedIndices.push_back(idx);
    } else {
      // Regular case: calculate the (paritial) flattened index
      auto multipliedValue = builder.create<arith::ConstantOp>(
          UnknownLoc::get(ctx),
          builder.getIntegerAttr(builder.getI64Type(), coeff));
      auto idx = builder.create<arith::MulIOp>(UnknownLoc::get(ctx),
                                               mlirIndexValue, multipliedValue);
      multipliedIndices.push_back(idx);
    }
  }
  // [END calculate the flattened array indices]

  // If we do not start from a memref type, then it must be from a chain of
  // GEPs. Here we accumulate our result onto that.
  if (auto pointerOperand = valueMap[gepInst->getPointerOperand()];
      pointerOperand && !isa<MemRefType>(pointerOperand.getType())) {
    multipliedIndices.push_back(valueMap[gepInst->getPointerOperand()]);
  }

  // [START accumulate the array index]
  // We construct a balanced tree of additions to accumulate the final index.
  // Build balanced tree
  std::function<mlir::Value(ArrayRef<mlir::Value>)> buildAdderTree =
      [&](ArrayRef<mlir::Value> vals) -> mlir::Value {
    assert(!vals.empty());
    if (vals.size() == 1)
      return vals[0];
    auto mid = vals.size() / 2;
    auto lhs = buildAdderTree(vals.take_front(mid));
    auto rhs = buildAdderTree(vals.drop_front(mid));
    return builder.create<arith::AddIOp>(UnknownLoc::get(ctx), lhs, rhs);
  };
  mlir::Value accumulatedArrayIndex = buildAdderTree(multipliedIndices);
  // [END accumulate the array index]

  valueMap[gepInst] = accumulatedArrayIndex;
}

void TranslateLLVMToStd::translateBranchInst(llvm::BranchInst *inst) {
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

void TranslateLLVMToStd::translateLoadInst(llvm::LoadInst *loadInst) {
  auto *instAddr = loadInst->getPointerOperand();
  mlir::Value memref;
  mlir::Value index = valueMap[loadInst->getPointerOperand()];
  mlir::Type resType = getMLIRType(loadInst->getType(), ctx);

  mlir::Value indexOp;

  if (getInstToMemRefMap.count(instAddr)) {
    memref = getInstToMemRefMap[instAddr];
    // LoadOp needs the index operand to be of index type
    indexOp = builder.create<arith::IndexCastOp>(UnknownLoc::get(ctx),
                                                 builder.getIndexType(), index);
  } else {
    assert(isa<MemRefType>(index.getType()));
    memref = index;
    indexOp = builder.create<arith::ConstantOp>(UnknownLoc::get(ctx),
                                                builder.getIndexAttr(0));
  }

  auto newOp =
      builder.create<memref::LoadOp>(UnknownLoc::get(ctx), resType, memref,
                                     /*indices = */ indexOp);
  valueMap[loadInst] = newOp.getResult();
  translateMemDepAndNameAttrs(loadInst, newOp, *ctx, builder);
}

void TranslateLLVMToStd::translateStoreInst(llvm::StoreInst *storeInst) {
  auto *instAddr = storeInst->getPointerOperand();
  mlir::Value memref;
  mlir::Value index = valueMap[storeInst->getPointerOperand()];

  mlir::Value indexOp;

  if (getInstToMemRefMap.count(instAddr)) {
    memref = getInstToMemRefMap[instAddr];
    // LoadOp needs the index operand to be of index type
    indexOp = builder.create<arith::IndexCastOp>(UnknownLoc::get(ctx),
                                                 builder.getIndexType(), index);
  } else {
    assert(isa<MemRefType>(index.getType()));
    memref = index;
    indexOp = builder.create<arith::ConstantOp>(UnknownLoc::get(ctx),
                                                builder.getIndexAttr(0));
  }

  mlir::Value storeValue = valueMap[storeInst->getValueOperand()];
  auto newOp =
      builder.create<memref::StoreOp>(UnknownLoc::get(ctx), storeValue, memref,
                                      /*indices = */ indexOp);
  translateMemDepAndNameAttrs(storeInst, newOp, *ctx, builder);
}

void TranslateLLVMToStd::translateAllocaInst(llvm::AllocaInst *allocaInst) {
  Location loc = UnknownLoc::get(ctx);

  // flatten the MD array into 1D
  int64_t arraySize = 1;
  llvm::Type *baseElementType = allocaInst->getAllocatedType();

  while (baseElementType->isArrayTy()) {
    arraySize *= baseElementType->getArrayNumElements();
    baseElementType = baseElementType->getArrayElementType();
  }

  assert(arraySize > 0 && "The size of the array must be positive!");

  auto memrefType = MemRefType::get(/*shape =*/{arraySize},
                                    getMLIRType(baseElementType, ctx));

  auto allocaOp = builder.create<memref::AllocaOp>(loc, memrefType);
  valueMap[allocaInst] = allocaOp->getResult(0);
}

void TranslateLLVMToStd::translateMemsetIntrinsic(llvm::CallInst *callInst) {
  Function *calledFunc = callInst->getCalledFunction();
  assert(calledFunc);
  assert(calledFunc->getIntrinsicID() == Intrinsic::memset);
  // -- Semantic of the memset intrinsic: --
  //
  // memset(dest : ptr, val : i8, length : i64, isVolatile : i1);
  // TODO: For now, we convert it to a set of stores; maybe in the future it
  // can be implemented using something smarter.
  mlir::Value memref;

  // We will treat dest as memref[offset], and store the specified values there
  // - When the ptr a function argument, the offset is zero.
  // - When the ptr is from a GEP, the offset is the value calculated from
  // there.
  mlir::Value offset;

  if (valueMap.count(callInst->getArgOperand(0)) &&
      isa_and_nonnull<MemRefType>(
          valueMap[callInst->getArgOperand(0)].getType())) {
    // Case: When the ptr operand is a function argument
    memref = valueMap[callInst->getArgOperand(0)];
    offset = builder.create<arith::ConstantOp>(
        UnknownLoc::get(ctx), IntegerAttr::get(builder.getIndexType(), 0));
  } else if (getInstToMemRefMap.count(callInst->getArgOperand(0))) {
    // Case: When the ptr operand is a GEP
    memref = getInstToMemRefMap[callInst->getArgOperand(0)];
    offset = valueMap[callInst->getArgOperand(0)];
  } else {
    // clang-format off
    LLVM_DEBUG(
        llvm::errs() << "Cannot determine the base ptr of memset!\n";
        llvm::errs() << "llvm value:\n";
        callInst->getArgOperand(0)->dump();
    );
    // clang-format on
    llvm::report_fatal_error(
        "Cannot determine the base ptr of the memset intrinsic!");
  }

  mlir::Value valToSet = valueMap[callInst->getArgOperand(1)];
  if (!valToSet || !isa<mlir::IntegerType>(valToSet.getType()))
    llvm::report_fatal_error(
        "Cannot determine the value to set of the memset intrinsic!");

  assert(valToSet.getType().getIntOrFloatBitWidth() == 8 &&
         "We assume that memset sets values of i8.");

  mlir::Value mlirLengthVal = valueMap[callInst->getArgOperand(2)];
  if (!mlirLengthVal || !isa<mlir::IntegerType>(mlirLengthVal.getType()))
    llvm::report_fatal_error(
        "Cannot determine the length to set of the memset intrinsic!");

  //
  unsigned length = 0;
  if (auto *constLength =
          llvm::dyn_cast<llvm::ConstantInt>(callInst->getArgOperand(2))) {
    length = constLength->getLimitedValue();
  } else {
    llvm::report_fatal_error(
        "Cannot determine the length to set of the memset intrinsic!");
  }

  // Determine:
  // - the value to be stored in the data type of the memref
  // - the number of elements
  auto memrefElemType =
      mlir::cast<MemRefType>(memref.getType()).getElementType();

  unsigned elemWidth = memrefElemType.getIntOrFloatBitWidth();

  // Say we store 1 (in i8) for length 64, with the element type i32
  // Here we need to store 64 * i8 / i32 = 16 elements
  int64_t valueInTargetType = 0;
  assert(elemWidth % 8 == 0 && "");
  if (auto *constInst =
          llvm::dyn_cast<llvm::ConstantInt>(callInst->getArgOperand(1))) {
    auto i8value = constInst->getLimitedValue();

    for (size_t bytePos = 0; bytePos < elemWidth / 8; ++bytePos) {
      valueInTargetType += i8value << (bytePos * 8);
    }
    unsigned numElemsToStore = length * 8 / elemWidth;

    auto constOp = builder.create<arith::ConstantIntOp>(
        UnknownLoc::get(ctx), valueInTargetType, elemWidth);

    for (size_t elemPos = 0; elemPos < numElemsToStore; ++elemPos) {
      auto constIdx = builder.create<arith::ConstantIntOp>(
          UnknownLoc::get(ctx), elemPos, offset.getType());
      // Add the constant op with the offset
      auto offsetPlusPos =
          builder.create<arith::AddIOp>(UnknownLoc::get(ctx), offset, constIdx);
      mlir::Value storeIndex = builder.create<arith::IndexCastOp>(
          UnknownLoc::get(ctx), builder.getIndexType(), offsetPlusPos);
      builder.create<memref::StoreOp>(UnknownLoc::get(ctx), constOp, memref,
                                      storeIndex);
    }
  }
}

void TranslateLLVMToStd::translateFunnelShiftIntrinsic(
    llvm::CallInst *callInst) {
  Function *calledFunc = callInst->getCalledFunction();
  assert(calledFunc);
  assert(calledFunc->getIntrinsicID() == Intrinsic::fshl ||
         calledFunc->getIntrinsicID() == Intrinsic::fshr);
  // -- Conversion for LLVM funnel shift intrinsic (llvm.fshl) --
  // Semantic:
  // r = fshl(a, b, c)
  // - concate: Wide = {a : b}
  // - shift to left: Wide <<= c
  // - take msb (same width as a and b)
  //
  // -- Conversion for LLVM funnel shift intrinsic (llvm.fshr) --
  // Semantic:
  // r = fshl(a, b, c)
  // - concate: Wide = {a : b}
  // - shift to right: Wide >>= c
  // - take lsb (same width as a and b)
  //
  // This block handles both.
  mlir::Value a = valueMap[callInst->getArgOperand(0)];
  mlir::Value b = valueMap[callInst->getArgOperand(1)];
  mlir::Value c = valueMap[callInst->getArgOperand(2)];

  unsigned width = a.getType().getIntOrFloatBitWidth();

  assert(a.getType().getIntOrFloatBitWidth() ==
         b.getType().getIntOrFloatBitWidth());

  // 1. Extend to width a + b

  auto aExt = builder.create<arith::ExtUIOp>(
      UnknownLoc::get(ctx), mlir::IntegerType::get(ctx, width * 2), a);
  auto bExt = builder.create<arith::ExtUIOp>(
      UnknownLoc::get(ctx), mlir::IntegerType::get(ctx, width * 2), b);
  auto cExt = builder.create<arith::ExtUIOp>(
      UnknownLoc::get(ctx), mlir::IntegerType::get(ctx, width * 2), c);

  auto constValueWidth = builder.create<arith::ConstantIntOp>(
      UnknownLoc::get(ctx), width, aExt.getType().getIntOrFloatBitWidth());

  // 2. Shift `a` and OR `b`
  auto aShift = builder.create<arith::ShLIOp>(UnknownLoc::get(ctx), aExt,
                                              constValueWidth);
  auto aConcatB =
      builder.create<arith::OrIOp>(UnknownLoc::get(ctx), aShift, bExt);

  // 3. Shift the wide value by c
  mlir::Value wideShifted;
  mlir::Value resultExt;

  if (calledFunc->getIntrinsicID() == Intrinsic::fshl) {
    wideShifted =
        builder.create<arith::ShLIOp>(UnknownLoc::get(ctx), aConcatB, cExt);
    // 4. Take the upper part.
    // - First shift the upper `width` bits to the lsb position
    resultExt = builder.create<arith::ShRUIOp>(UnknownLoc::get(ctx),
                                               wideShifted, constValueWidth);
  } else {
    resultExt =
        builder.create<arith::ShRUIOp>(UnknownLoc::get(ctx), aConcatB, cExt);
  }

  // - Then truncate the result down to `width`-bit
  auto result = builder.create<arith::TruncIOp>(
      UnknownLoc::get(ctx), mlir::IntegerType::get(ctx, width), resultExt);

  valueMap[callInst] = result;
}

void TranslateLLVMToStd::translateCallInst(llvm::CallInst *callInst) {

  Function *calledFunc = callInst->getCalledFunction();
  assert(calledFunc);
  assert(calledFunc->isIntrinsic() &&
         "Function calls are not currently supported");

  if (calledFunc->getIntrinsicID() == Intrinsic::smax) {
    mlir::Value lhs = valueMap[callInst->getArgOperand(0)];
    mlir::Value rhs = valueMap[callInst->getArgOperand(1)];
    auto retType = getMLIRType(callInst->getType(), ctx);
    naiveTranslation<arith::MaxSIOp>(retType, {lhs, rhs}, callInst);
  } else if (calledFunc->getIntrinsicID() == Intrinsic::fabs) {
    mlir::Value arg = valueMap[callInst->getArgOperand(0)];
    auto retType = getMLIRType(callInst->getType(), ctx);
    naiveTranslation<math::AbsFOp>(retType, {arg}, callInst);
  } else if (calledFunc->getIntrinsicID() == Intrinsic::memset) {
    this->translateMemsetIntrinsic(callInst);
  } else if (calledFunc->getIntrinsicID() == Intrinsic::fshl ||
             calledFunc->getIntrinsicID() == Intrinsic::fshr) {
    this->translateFunnelShiftIntrinsic(callInst);
  } else {
    LLVM_DEBUG(llvm::errs() << "Unhandled intrinsic:"; callInst->dump(););
    llvm::report_fatal_error(
        "Not implemented llvm intrinsic function handling!");
  }
}
