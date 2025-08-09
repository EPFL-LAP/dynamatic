#include "ImportLLVMModule.h"

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
    llvmType->dump();
    assert(false);
  }
  return mlirType;
}

void ImportLLVMModule::createConstants(llvm::Function *llvmFunc,
                                       OpBuilder &builder, Location loc) {
  for (auto &block : *llvmFunc) {
    for (auto &inst : block) {
      for (unsigned i = 0; i < inst.getNumOperands(); ++i) {
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
        }

        if (auto *floatConst = dyn_cast<llvm::ConstantFP>(val)) {
          const APFloat &floatVal = floatConst->getValue();
          if (&floatVal.getSemantics() == &llvm::APFloat::IEEEsingle()) {
            auto constOp = builder.create<arith::ConstantFloatOp>(
                loc, floatVal, builder.getF32Type());
            valueMapping[val] = constOp->getResult(0);
          } else if (&floatVal.getSemantics() == &llvm::APFloat::IEEEdouble()) {
            auto constOp = builder.create<arith::ConstantFloatOp>(
                loc, floatVal, builder.getF64Type());
            valueMapping[val] = constOp->getResult(0);
          }
        }
      }
    }
  }
}

void ImportLLVMModule::translateOperation(llvm::Instruction *inst,
                                          Location &loc) {
  inst->dump();
  assert(converted.count(inst) == 0);

  SmallVector<mlir::Value> mlirOpOperands;
  for (unsigned i = 0; i < inst->getNumOperands(); ++i) {
    llvm::Value *val = inst->getOperand(i);
    if (!valueMapping.count(val)) {
      inst->dump();
      llvm_unreachable("Missing mapping from LLVM value to MLIR value");
    }
    mlirOpOperands.push_back(valueMapping[val]);
  }

  if (auto *binaryOp = dyn_cast<llvm::BinaryOperator>(inst)) {
    mlir::Type resType = convertLLVMTypeToMLIR(inst->getType(), ctx);
    switch (binaryOp->getOpcode()) {
      // clang-format off
        case Instruction::Add: translateBinaryOp<arith::AddIOp>(builder, loc, resType, mlirOpOperands, inst); break;
        case Instruction::Mul: translateBinaryOp<arith::MulIOp>(builder, loc, resType, mlirOpOperands, inst); break;
        case Instruction::Shl: translateBinaryOp<arith::ShLIOp>(builder, loc, resType, mlirOpOperands, inst); break;
        case Instruction::AShr: translateBinaryOp<arith::ShRSIOp>(builder, loc, resType, mlirOpOperands, inst); break;
        case Instruction::LShr: translateBinaryOp<arith::ShRUIOp>(builder, loc, resType, mlirOpOperands, inst); break;
      // clang-format on
    default: {
      llvm_unreachable("Not implemented");
    }
    }

    converted.insert(inst);
  } else if (auto *icmpInst = dyn_cast<llvm::ICmpInst>(inst)) {
    arith::CmpIPredicate pred;
    switch (icmpInst->getPredicate()) {
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
      // clang-format on
    default:
      // Handle unknown predicate or assert
      llvm_unreachable("Unsupported ICMP predicate");
    }

    auto op = builder.create<arith::CmpIOp>(loc, pred, mlirOpOperands[0],
                                            mlirOpOperands[1]);
    addMapping(inst, op.getResult());
    loc = op.getLoc();
  } else if (auto *selInst = dyn_cast<llvm::SelectInst>(inst)) {
    mlir::Type resType = convertLLVMTypeToMLIR(inst->getType(), ctx);
    translateBinaryOp<arith::SelectOp>(builder, loc, resType, mlirOpOperands,
                                       inst);
  } else if (auto *branchInst = dyn_cast<llvm::BranchInst>(inst)) {

  }

  else if (auto *returnOp = dyn_cast<llvm::ReturnInst>(inst)) {
    builder.create<func::ReturnOp>(loc, mlirOpOperands);
  }

  else {
    llvm_unreachable("Not implemented");
  }
}

void ImportLLVMModule::translateLLVMFunction(llvm::Function *llvmFunc) {
  builder.setInsertionPointToEnd(mlirModule->getBlock());
  SmallVector<mlir::Type> argTypes;
  for (auto &arg : llvmFunc->args()) {
    argTypes.push_back(
        convertLLVMTypeToMLIR(arg.getType(), ctx)); // Simplification
  }

  auto funcType = builder.getFunctionType(argTypes, {builder.getI32Type()});
  auto funcOp = builder.create<func::FuncOp>(mlirModule->getLoc(),
                                             llvmFunc->getName(), funcType);

  funcOp.dump();

  llvm::errs() << "LLVM IR, # arguments" << llvmFunc->arg_size() << "\n";
  llvm::errs() << "MLIR IR, # arguments" << funcOp.getNumArguments() << "\n";

  // Convert the entry block
  auto &entryBlock = *funcOp.addEntryBlock();

  // Note: funcType and the actual function arguments (i.e., the arguments of
  // the first block) are two separate concepts:
  for (auto &arg : llvmFunc->args()) {
    entryBlock.addArgument(convertLLVMTypeToMLIR(arg.getType(), ctx),
                           funcOp->getLoc());
  }

  for (unsigned i = 0; i < llvmFunc->arg_size(); i++) {
    llvm::errs() << "Arg # " << i << "\n";
    valueMapping[llvmFunc->getArg(i)] = funcOp.getArgument(i);
  }

  builder.setInsertionPointToStart(&entryBlock);

  Location loc = mlir::UnknownLoc::get(ctx);
  createConstants(llvmFunc, builder, loc);
  for (auto &inst : llvmFunc->getEntryBlock()) {
    translateOperation(&inst, loc);
  }

  mlirModule.push_back(funcOp);
}