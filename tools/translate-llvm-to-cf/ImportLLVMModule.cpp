#include "ImportLLVMModule.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

mlir::Type convertLLVMTypeToMLIR(llvm::Type *llvmType,
                                 mlir::MLIRContext *context) {
  mlir::Type mlirType;

  if (llvmType->isIntegerTy()) {
    mlirType = mlir::IntegerType::get(context, llvmType->getIntegerBitWidth());
  } else if (llvmType->isFloatTy()) {
    mlirType = mlir::FloatType::getF32(context);
  } else if (llvmType->isDoubleTy()) {
    mlirType = mlir::FloatType::getF64(context);
  } else {
    llvmType->dump();
    assert(false);
  }
  return mlirType;
}

SmallVector<mlir::Value>
ImportLLVMModule::getBranchOperandsForCFGEdge(BasicBlock *currentBB,
                                              BasicBlock *nextBB) {
  SmallVector<mlir::Value> operands;
  for (PHINode &phi : nextBB->phis()) {
    mlir::Value argument =
        valueMapping[phi.getIncomingValueForBlock(currentBB)];
    operands.push_back(argument);
  }
  return operands;
}

void ImportLLVMModule::initializeBlocksAndBlockMapping(llvm::Function *llvmFunc,
                                                       func::FuncOp funcOp) {

  // Convert the entry block

  // NOTE: this automatically adds the block arguments of the first block
  // according to the predefined function signiture.
  Block *entryBlock = funcOp.addEntryBlock();
  BasicBlock &entryBB = llvmFunc->getEntryBlock();
  blockMapping[&entryBB] = entryBlock;

  // Note: funcType and the actual function arguments (i.e., the arguments of
  // the first block) are two separate concepts:

  for (auto [llvmArg, mlirArg] :
       llvm::zip_equal(llvmFunc->args(), entryBlock->getArguments())) {
    valueMapping[&llvmArg] = mlirArg;
  }

  // Remaining basic blocks
  for (BasicBlock &bb : *llvmFunc) {

    if (&bb == &entryBB)
      continue;

    auto *block = funcOp.addBlock();
    assert(!blockMapping.count(&bb));
    blockMapping[&bb] = block;
    for (auto &phi : bb.phis()) {
      auto mlirArg =
          block->addArgument(convertLLVMTypeToMLIR(phi.getType(), ctx),
                             mlir::UnknownLoc::get(ctx));
      valueMapping[&phi] = mlirArg;
      converted.insert(&phi);
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
        if (!isa<llvm::Constant>(val) || valueMapping.count(val)) {
          continue;
        }

        if (auto *intConst = dyn_cast<ConstantInt>(val)) {
          APInt intVal = intConst->getValue();
          auto constOp = builder.create<arith::ConstantIntOp>(
              loc, intVal.getSExtValue(), intVal.getBitWidth());
          valueMapping[val] = constOp->getResult(0);
          loc = constOp->getLoc();
        }

        if (auto *floatConst = dyn_cast<llvm::ConstantFP>(val)) {
          const APFloat &floatVal = floatConst->getValue();
          if (&floatVal.getSemantics() == &llvm::APFloat::IEEEsingle()) {
            auto constOp = builder.create<arith::ConstantFloatOp>(
                loc, floatVal, builder.getF32Type());
            valueMapping[val] = constOp->getResult(0);
            loc = constOp->getLoc();
          } else if (&floatVal.getSemantics() == &llvm::APFloat::IEEEdouble()) {
            auto constOp = builder.create<arith::ConstantFloatOp>(
                loc, floatVal, builder.getF64Type());
            valueMapping[val] = constOp->getResult(0);
            loc = constOp->getLoc();
          }
        }
      }
    }
  }
}

void ImportLLVMModule::translateBinaryInst(llvm::BinaryOperator *inst) {
  Location loc = UnknownLoc::get(ctx);
  mlir::Value lhs = valueMapping[inst->getOperand(0)];
  mlir::Value rhs = valueMapping[inst->getOperand(1)];
  mlir::Type resType = convertLLVMTypeToMLIR(inst->getType(), ctx);
  switch (inst->getOpcode()) {
    // clang-format off
    case Instruction::Add:  naiveTranslation<arith::AddIOp>( loc, resType,  {lhs, rhs}, inst); break;
    case Instruction::Sub:  naiveTranslation<arith::SubIOp>( loc, resType,  {lhs, rhs}, inst); break;
    case Instruction::Mul:  naiveTranslation<arith::MulIOp>( loc, resType,  {lhs, rhs}, inst); break;
    case Instruction::UDiv: naiveTranslation<arith::DivUIOp>( loc, resType,  {lhs, rhs}, inst); break;
    case Instruction::SDiv: naiveTranslation<arith::DivSIOp>( loc, resType,  {lhs, rhs}, inst); break;
    case Instruction::SRem: naiveTranslation<arith::RemSIOp>( loc, resType,  {lhs, rhs}, inst); break;
    case Instruction::URem: naiveTranslation<arith::RemUIOp>( loc, resType,  {lhs, rhs}, inst); break;
    case Instruction::FAdd: naiveTranslation<arith::AddFOp>( loc, resType,  {lhs, rhs}, inst); break;
    case Instruction::FSub: naiveTranslation<arith::SubFOp>( loc, resType,  {lhs, rhs}, inst); break;
    case Instruction::FDiv: naiveTranslation<arith::DivFOp>( loc, resType,  {lhs, rhs}, inst); break;
    case Instruction::FMul: naiveTranslation<arith::MulFOp>( loc, resType,  {lhs, rhs}, inst); break;
    case Instruction::Shl:  naiveTranslation<arith::ShLIOp>( loc, resType,  {lhs, rhs}, inst); break;
    case Instruction::AShr: naiveTranslation<arith::ShRSIOp>( loc, resType, {lhs, rhs}, inst); break;
    case Instruction::LShr: naiveTranslation<arith::ShRUIOp>( loc, resType, {lhs, rhs}, inst); break;
    case Instruction::And:  naiveTranslation<arith::AndIOp>( loc, resType, {lhs, rhs}, inst); break;
    case Instruction::Or:   naiveTranslation<arith::OrIOp>( loc, resType, {lhs, rhs}, inst); break;
    case Instruction::Xor:  naiveTranslation<arith::XOrIOp>( loc, resType, {lhs, rhs}, inst); break;
    // clang-format on
  default: {
    llvm::errs() << "Not yet handled binary operation type "
                 << inst->getOpcodeName() << "\n";
    llvm_unreachable("Not handled operation type");
  }
  }
}

void ImportLLVMModule::translateCastInst(llvm::CastInst *inst) {
  Location loc = UnknownLoc::get(ctx);
  mlir::Value arg = valueMapping[inst->getOperand(0)];
  mlir::Type resType = convertLLVMTypeToMLIR(inst->getType(), ctx);

  switch (inst->getOpcode()) {
    // clang-format off
    case Instruction::ZExt: naiveTranslation<arith::ExtUIOp>(loc, resType, {arg}, inst); break;
    case Instruction::SExt: naiveTranslation<arith::ExtSIOp>(loc, resType, {arg}, inst); break;
    case Instruction::FPExt: naiveTranslation<arith::ExtFOp>(loc, resType, {arg}, inst); break;
    case Instruction::Trunc: naiveTranslation<arith::TruncIOp>(loc, resType, {arg}, inst); break;
    case Instruction::FPTrunc: naiveTranslation<arith::TruncFOp>(loc, resType, {arg}, inst); break;
    case Instruction::SIToFP: naiveTranslation<arith::SIToFPOp>(loc, resType, {arg}, inst); break;
    case Instruction::FPToSI: naiveTranslation<arith::FPToSIOp>(loc, resType, {arg}, inst); break;
    case Instruction::FPToUI: naiveTranslation<arith::FPToUIOp>(loc, resType, {arg}, inst); break;
    // clang-format on
  default: {
    llvm::errs() << "Not yet handled binary operation type "
                 << inst->getOpcodeName() << "\n";
    llvm_unreachable("Not implemented");
  }
  }
}

void ImportLLVMModule::translateICmpInst(llvm::ICmpInst *inst) {
  Location loc = UnknownLoc::get(ctx);

  mlir::Value lhs = valueMapping[inst->getOperand(0)];
  mlir::Value rhs = valueMapping[inst->getOperand(1)];
  arith::CmpIPredicate pred;
  switch (inst->getPredicate()) {
    // clang-format off
    case llvm::CmpInst::Predicate::ICMP_EQ:  pred = arith::CmpIPredicate::eq;  break;
    case llvm::CmpInst::Predicate::ICMP_NE:  pred = arith::CmpIPredicate::ne;  break;
    case llvm::CmpInst::Predicate::ICMP_UGT: pred = arith::CmpIPredicate::ugt; break;
    case llvm::CmpInst::Predicate::ICMP_UGE: pred = arith::CmpIPredicate::uge; break;
    case llvm::CmpInst::Predicate::ICMP_ULT: pred = arith::CmpIPredicate::ult; break;
    case llvm::CmpInst::Predicate::ICMP_ULE: pred = arith::CmpIPredicate::ule; break;
    case llvm::CmpInst::Predicate::ICMP_SGT: pred = arith::CmpIPredicate::sgt; break;
    case llvm::CmpInst::Predicate::ICMP_SGE: pred = arith::CmpIPredicate::sge; break;
    case llvm::CmpInst::Predicate::ICMP_SLT: pred = arith::CmpIPredicate::slt; break;
    case llvm::CmpInst::Predicate::ICMP_SLE: pred = arith::CmpIPredicate::sle; break;

    default: llvm_unreachable("Unsupported ICMP predicate");
    // clang-format on
  }

  auto op = builder.create<arith::CmpIOp>(loc, pred, lhs, rhs);
  addMapping(inst, op.getResult());
  loc = op.getLoc();
}

void ImportLLVMModule::translateFCmpInst(llvm::FCmpInst *inst) {

  Location loc = UnknownLoc::get(ctx);
  mlir::Value lhs = valueMapping[inst->getOperand(0)];
  mlir::Value rhs = valueMapping[inst->getOperand(1)];
  arith::CmpFPredicate pred;

  switch (inst->getPredicate()) {
    // clang-format off
    // Ordered comparisons
    case llvm::CmpInst::FCMP_OEQ: pred = arith::CmpFPredicate::OEQ; break;
    case llvm::CmpInst::FCMP_OGT: pred = arith::CmpFPredicate::OGT; break;
    case llvm::CmpInst::FCMP_OGE: pred = arith::CmpFPredicate::OGE; break;
    case llvm::CmpInst::FCMP_OLT: pred = arith::CmpFPredicate::OLT; break;
    case llvm::CmpInst::FCMP_OLE: pred = arith::CmpFPredicate::OLE; break;
    case llvm::CmpInst::FCMP_ONE: pred = arith::CmpFPredicate::ONE; break;

    // Ordered / unordered special checks
    case llvm::CmpInst::FCMP_ORD: pred = arith::CmpFPredicate::ORD; break;
    case llvm::CmpInst::FCMP_UNO: pred = arith::CmpFPredicate::UNO; break;

    // Unordered comparisons
    case llvm::CmpInst::FCMP_UEQ: pred = arith::CmpFPredicate::UEQ; break;
    case llvm::CmpInst::FCMP_UGT: pred = arith::CmpFPredicate::UGT; break;
    case llvm::CmpInst::FCMP_UGE: pred = arith::CmpFPredicate::UGE; break;
    case llvm::CmpInst::FCMP_ULT: pred = arith::CmpFPredicate::ULT; break;
    case llvm::CmpInst::FCMP_ULE: pred = arith::CmpFPredicate::ULE; break;
    case llvm::CmpInst::FCMP_UNE: pred = arith::CmpFPredicate::UNE; break;

    // No comparison (always false/true)
    case llvm::CmpInst::FCMP_FALSE: pred = arith::CmpFPredicate::AlwaysFalse; break;
    case llvm::CmpInst::FCMP_TRUE:  pred = arith::CmpFPredicate::AlwaysTrue; break;

    default: llvm_unreachable("Unsupported FCMP predicate");
    // clang-format on
  }
  auto op = builder.create<arith::CmpFOp>(loc, pred, lhs, rhs);
  addMapping(inst, op.getResult());
  loc = op.getLoc();
}

void ImportLLVMModule::translateGEPOp(llvm::GetElementPtrInst *gepInst) {
  // Check if the GEP is not chained
  mlir::Value baseAddress = valueMapping[gepInst->getPointerOperand()];
  SmallVector<mlir::Value, 4> indexOperands;

  auto memrefType = baseAddress.getType().dyn_cast<MemRefType>();

  if (!memrefType)
    llvm_unreachable("GEP should take memref as reference");

  for (auto &indexUse : gepInst->indices()) {
    llvm::Value *indexValue = indexUse;
    mlir::Value mlirIndexValue = valueMapping[indexValue];
    // NOTE: memref::LoadOp and memref::StoreOp expect their indices to be of
    // IndexType. Therefore, we cast the i32/i64 indices to IndexType. This
    // pattern will later be folded in the bitwidth optimization pass.
    auto idxCastOp = builder.create<arith::IndexCastOp>(
        UnknownLoc::get(ctx), builder.getIndexType(), mlirIndexValue);
    indexOperands.push_back(idxCastOp);
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
  int remainingConstZeros =
      memrefType.getShape().size() - gepInst->getNumIndices();
  assert(remainingConstZeros >= 0 &&
         "GEP should only omit indices, but shouldn't have more indices than "
         "the original memref type extracted from the function argument!");
  for (int i = 0; i < remainingConstZeros; i++) {
    auto constZeroOp = builder.create<arith::ConstantOp>(
        UnknownLoc::get(ctx), builder.getI64IntegerAttr(0));
    auto idxCastOp = builder.create<arith::IndexCastOp>(
        UnknownLoc::get(ctx), builder.getIndexType(), constZeroOp);
    indexOperands.push_back(idxCastOp);
  }

  for (auto *user : gepInst->users()) {
    if (isa<GetElementPtrInst>(user)) {
      llvm_unreachable(
          "Invalid GEP -> GEP pattern detected! It is assumed that this kind "
          "of pattern is canonicalized away using instcombine pass.");
    } else if (auto *loadInst = dyn_cast<LoadInst>(user)) {
      auto newLoadOp = builder.create<memref::LoadOp>(
          UnknownLoc::get(ctx), convertLLVMTypeToMLIR(loadInst->getType(), ctx),
          baseAddress, ValueRange(indexOperands));
      valueMapping[user] = newLoadOp.getResult();
      converted.insert(loadInst);
    } else if (auto *storeInst = dyn_cast<StoreInst>(user)) {

      auto storeValue = valueMapping[storeInst->getValueOperand()];
      builder.create<memref::StoreOp>(UnknownLoc::get(ctx), storeValue,
                                      baseAddress, indexOperands);
      converted.insert(storeInst);
    }
  }
}

void ImportLLVMModule::translateBranchInst(llvm::BranchInst *inst) {
  BasicBlock *currLLVMBB = inst->getParent();
  Location loc = UnknownLoc::get(ctx);
  if (inst->isUnconditional()) {
    BasicBlock *nextLLVMBB = dyn_cast_or_null<BasicBlock>(inst->getOperand(0));
    assert(nextLLVMBB &&
           "The unconditional branch doesn't have a BB as operand!");
    auto op = builder.create<cf::BranchOp>(
        loc, blockMapping[nextLLVMBB],
        getBranchOperandsForCFGEdge(currLLVMBB, nextLLVMBB));
    loc = op.getLoc();
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

    mlir::Value condition = valueMapping[inst->getCondition()];

    auto op = builder.create<cf::CondBranchOp>(
        loc, condition, blockMapping[trueDestBB], trueOperands,
        blockMapping[falseDestBB], falseOperands);
    loc = op.getLoc();
  }
}

void ImportLLVMModule::translateLoadWithZeroIndices(llvm::LoadInst *loadInst) {
  Location loc = UnknownLoc::get(ctx);
  // NOTE: This condition handles a special case where a load only has
  // constant indices, e.g., tmp = mat[0][0].
  auto *instAddr = loadInst->getPointerOperand();
  if (isa<GetElementPtrInst>(instAddr))
    llvm_unreachable(
        "Converting a load but the producer hasn't been converted yet!");
  mlir::Value addressVal = valueMapping[instAddr];
  auto memrefType = addressVal.getType().dyn_cast<MemRefType>();

  SmallVector<mlir::Value> indexValues;
  int constZerosToAdd = memrefType.getShape().size();
  for (int i = 0; i < constZerosToAdd; i++) {
    auto constZeroOp =
        builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(0));
    indexValues.push_back(constZeroOp);
  }
  mlir::Type resType = convertLLVMTypeToMLIR(loadInst->getType(), ctx);
  auto newOp =
      builder.create<memref::LoadOp>(loc, resType, addressVal, indexValues);
  valueMapping[loadInst] = newOp.getResult();
}

void ImportLLVMModule::translateStoreWithZeroIndices(
    llvm::StoreInst *storeInst) {
  Location loc = UnknownLoc::get(ctx);
  // NOTE: This condition handles a special case where a load only has
  // constant indices, e.g., tmp = mat[0][0].
  auto *instAddr = storeInst->getPointerOperand();
  if (isa<GetElementPtrInst>(instAddr))
    llvm_unreachable(
        "Converting a load but the producer hasn't been converted yet!");
  mlir::Value addressVal = valueMapping[instAddr];
  auto memrefType = addressVal.getType().dyn_cast<MemRefType>();

  SmallVector<mlir::Value> indexValues;
  int constZerosToAdd = memrefType.getShape().size();
  for (int i = 0; i < constZerosToAdd; i++) {
    auto constZeroOp =
        builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(0));
    indexValues.push_back(constZeroOp);
  }
  mlir::Value storeValue = valueMapping[storeInst->getValueOperand()];

  builder.create<memref::StoreOp>(loc, storeValue, addressVal, indexValues);
}

void ImportLLVMModule::translateOperation(llvm::Instruction *inst) {
  inst->dump();

  Location loc = UnknownLoc::get(ctx);

  if (converted.count(inst)) {
    return;
  }

  if (auto *binaryOp = dyn_cast<llvm::BinaryOperator>(inst)) {
    translateBinaryInst(binaryOp);
  } else if (auto *castOp = dyn_cast<llvm::CastInst>(inst)) {
    translateCastInst(castOp);
  } else if (auto *gepInst = dyn_cast<llvm::GetElementPtrInst>(inst)) {
    translateGEPOp(gepInst);
  } else if (auto *loadInst = dyn_cast<llvm::LoadInst>(inst)) {
    translateLoadWithZeroIndices(loadInst);
  } else if (auto *storeInst = dyn_cast<llvm::StoreInst>(inst)) {
    translateStoreWithZeroIndices(storeInst);
  } else if (auto *icmpInst = dyn_cast<llvm::ICmpInst>(inst)) {
    translateICmpInst(icmpInst);
  } else if (auto *fcmpInst = dyn_cast<FCmpInst>(inst)) {
    translateFCmpInst(fcmpInst);
  } else if (auto *selInst = dyn_cast<llvm::SelectInst>(inst)) {
    mlir::Value condition = valueMapping[inst->getOperand(0)];
    mlir::Value trueOperand = valueMapping[inst->getOperand(1)];
    mlir::Value falseOperand = valueMapping[inst->getOperand(2)];
    mlir::Type resType = convertLLVMTypeToMLIR(inst->getType(), ctx);
    naiveTranslation<arith::SelectOp>(
        loc, resType, {condition, trueOperand, falseOperand}, inst);
  } else if (auto *branchInst = dyn_cast<llvm::BranchInst>(inst)) {
    translateBranchInst(branchInst);
  } else if (auto *returnOp = dyn_cast<llvm::ReturnInst>(inst)) {
    if (returnOp->getNumOperands() == 1) {
      mlir::Value arg = valueMapping[inst->getOperand(0)];
      builder.create<func::ReturnOp>(loc, arg);
    } else {
      builder.create<func::ReturnOp>(loc);
    }
  } else {
    llvm_unreachable("Not implemented");
  }

  // This method usually converts a single instruction. When it does DAG-DAG
  // conversion (e.g., GEP -> LOAD/STORE), we register the additionally
  // converted instructions in specific cases.
  converted.insert(inst);
}

void ImportLLVMModule::translateLLVMFunction(llvm::Function *llvmFunc) {

  SmallVector<mlir::Type> argTypes =
      getFuncArgTypes(llvmFunc->getName().str(), argMap, builder);

  // SmallVector<mlir::Type> argTypes;
  // for (auto &arg : llvmFunc->args()) {
  //   argTypes.push_back(convertLLVMTypeToMLIR(arg.getType(), ctx));
  // }

  llvm::errs() << "Available dialects:\n";

  for (auto str : ctx->getAvailableDialects()) {
    llvm::errs() << str << "\n";
  }

  builder.setInsertionPointToEnd(mlirModule.getBody());

  auto funcType = builder.getFunctionType(argTypes, {builder.getI32Type()});
  auto funcOp = builder.create<func::FuncOp>(builder.getUnknownLoc(),
                                             llvmFunc->getName(), funcType);

  funcOp.dump();

  initializeBlocksAndBlockMapping(llvmFunc, funcOp);
  llvm::errs() << "LLVM IR, # arguments" << llvmFunc->arg_size() << "\n";
  llvm::errs() << "MLIR IR, # arguments" << funcOp.getNumArguments() << "\n";
  llvm::errs() << "MLIR IR, # arguments block"
               << funcOp.getBlocks().front().getNumArguments() << "\n";

  auto &entryBlock = funcOp.getBody().front();
  builder.setInsertionPointToStart(&entryBlock);

  createConstants(llvmFunc);

  for (auto &block : *llvmFunc) {
    builder.setInsertionPointToEnd(blockMapping[&block]);
    for (auto &inst : block) {
      translateOperation(&inst);
    }
  }
}