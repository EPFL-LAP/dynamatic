//===- elastic-miter.cpp - The elastic-miter driver -------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the elastic-miter tool, it creates an elastic miter
// circuit, which can later be used to formally verify equivalence of two
// handshake circuits.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"

#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/InitAllDialects.h"

namespace cl = llvm::cl;
using namespace mlir;
using namespace dynamatic::handshake;

// TODO remove this
OpPrintingFlags printingFlags;

// Command line TODO actually use arguments
static cl::OptionCategory mainCategory("elastic-miter Options");

// TODO make not ugly
static SmallVector<NamedAttribute> createHandshakeAttr(OpBuilder &builder, int bb, const std::string &name) {
  MLIRContext *context = builder.getContext();
  auto i32 = IntegerType::get(context, 32);
  auto intAttr = IntegerAttr::get(i32, bb);
  auto nameAttr = builder.getNamedAttr("handshake.name", builder.getStringAttr(name));
  auto bbAttr = builder.getNamedAttr("handshake.bb", intAttr);
  SmallVector<NamedAttribute> attrs = {nameAttr, bbAttr};
  return attrs;
}

static LogicalResult changeHandshakeBB(MLIRContext &context, Operation *op, int bb) {
  auto i32 = IntegerType::get(&context, 32);
  auto attr = IntegerAttr::get(i32, bb);
  op->setAttr("handshake.bb", attr);
  return success();
}

// TODO documentation
static LogicalResult prefixOperation(Operation &op, const std::string &prefix) {
  StringAttr nameAttr = op.getAttrOfType<StringAttr>("handshake.name");
  if (!nameAttr)
    return failure();

  std::string newName = prefix + nameAttr.getValue().str();

  op.setAttr("handshake.name", StringAttr::get(op.getContext(), newName));

  return success();
}

// TODO clean this up, documentation, do not pass newBlock
static FuncOp buildNewFunc(OpBuilder builder, Block *newBlock, FuncOp &lhsFuncOp, FuncOp &rhsFuncOp) {
  // Copy the Result Names from lhs but prefix it with EQ_
  SmallVector<Attribute> prefixedResAttr;
  for (Attribute attr : lhsFuncOp.getResNames()) {
    auto strAttr = attr.dyn_cast<StringAttr>();
    if (strAttr) {
      prefixedResAttr.push_back(builder.getStringAttr("EQ_" + strAttr.getValue().str()));
    }
  }

  // TODO check equality of interfaces

  auto argNamedAttr = builder.getNamedAttr("argNames", lhsFuncOp.getArgNames());
  auto resNamedAttr = builder.getNamedAttr("resNames", builder.getArrayAttr(prefixedResAttr));

  ArrayRef<NamedAttribute> funcAttr({argNamedAttr, resNamedAttr});

  int outputSize = lhsFuncOp.getResultTypes().size();

  ChannelType i1ChannelType = ChannelType::get(builder.getI1Type());
  llvm::SmallVector<Type> outputTypes(outputSize, i1ChannelType); // replace all outputs with i1 channels which
                                                                  // represent the result of the comparison
  FunctionType funcType = builder.getFunctionType(lhsFuncOp.getArgumentTypes(), outputTypes);

  // Create the miter function
  FuncOp newFuncOp = builder.create<FuncOp>(builder.getUnknownLoc(), "elastic_miter", funcType, funcAttr);

  // Create a block and put it in the funcOp, this is borrowed from
  // func::funcOp.addEntryBlock()
  newFuncOp.push_back(newBlock);

  builder.setInsertionPointToStart(newBlock);

  // FIXME: Allow for passing in locations for these arguments instead of using
  // the operations location.
  ArrayRef<Type> inputTypes = newFuncOp.getArgumentTypes();
  SmallVector<Location> locations(inputTypes.size(), newFuncOp.getOperation()->getLoc());
  newBlock->addArguments(inputTypes, locations);

  return newFuncOp;
}

// create the elatic miter module, given two circuits
static FailureOr<OwningOpRef<ModuleOp>> createElasticMiter(MLIRContext &context) {

  // TODO use arguments
  llvm::StringRef filename("../tools/elastic-miter/rewrites/a_lhs.mlir");
  llvm::StringRef filename2("../tools/elastic-miter/rewrites/a_rhs.mlir");
  OwningOpRef<ModuleOp> lhsModule = parseSourceFile<ModuleOp>(filename, &context);
  OwningOpRef<ModuleOp> rhsModule = parseSourceFile<ModuleOp>(filename2, &context);

  // The module can only have one function so we just take the first element
  // TODO add a check for this
  FuncOp lhsFuncOp = *lhsModule->getOps<FuncOp>().begin();
  FuncOp rhsFuncOp = *rhsModule->getOps<FuncOp>().begin();

  OpBuilder builder(&context);

  OwningOpRef<ModuleOp> miterModule = ModuleOp::create(builder.getUnknownLoc());

  // Create a new function
  Block *newBlock = new Block();
  FuncOp newFuncOp = buildNewFunc(builder, newBlock, lhsFuncOp, rhsFuncOp);

  // Add the function to the module
  miterModule->push_back(newFuncOp);

  /* TODO:
  3: Add ND wire operations
  */

  // Rename the operations in the existing lhs module TODO check for success
  for (Operation &op : lhsFuncOp.getOps()) {
    prefixOperation(op, "lhs_");
  }
  // Rename the operations in the existing rhs module TODO check for success
  for (Operation &op : rhsFuncOp.getOps()) {
    prefixOperation(op, "rhs_");
  }

  builder.setInsertionPointToStart(newBlock);

  // TODO improve locations
  ForkOp forkOp;
  BufferOp lhsBufferOp;
  BufferOp rhsBufferOp;
  NDWireOp lhsNDWireOp;
  NDWireOp rhsNDWireOp;
  for (unsigned i = 0; i < lhsFuncOp.getNumArguments(); ++i) {
    BlockArgument lhsArgs = lhsFuncOp.getArgument(i);
    BlockArgument rhsArgs = rhsFuncOp.getArgument(i);
    BlockArgument miterArgs = newFuncOp.getArgument(i);

    TypeRange tyRange(miterArgs.getType());
    SmallVector<NamedAttribute> forkAttrs = createHandshakeAttr(builder, 0, "fork_" + lhsFuncOp.getArgName(i).str());
    SmallVector<NamedAttribute> bufAttrs = createHandshakeAttr(builder, 0, "buffer_" + lhsFuncOp.getArgName(i).str());

    // forkOp = builder.create<ForkOp>(newFuncOp.getLoc(), tyRange, miterArgs, forkAttrs);
    forkOp = builder.create<ForkOp>(newFuncOp.getLoc(), miterArgs, 2);
    lhsBufferOp = builder.create<BufferOp>(forkOp.getLoc(), forkOp.getResults()[0], TimingInfo::oehb(), 3);
    rhsBufferOp = builder.create<BufferOp>(forkOp.getLoc(), forkOp.getResults()[1], TimingInfo::oehb(), 3);
    lhsNDWireOp = builder.create<NDWireOp>(forkOp.getLoc(), lhsBufferOp.getResult());
    rhsNDWireOp = builder.create<NDWireOp>(forkOp.getLoc(), rhsBufferOp.getResult());

    // Use the newly created fork's output instead of the origial argument in
    // the lhsFuncOp's operations
    for (Operation *op : llvm::make_early_inc_range(lhsArgs.getUsers())) {
      op->replaceUsesOfWith(lhsArgs, lhsNDWireOp.getResult());
    }
    // Use the newly created fork's output instead of the origial argument in
    // the rhsFuncOp's operations
    for (Operation *op : llvm::make_early_inc_range(rhsArgs.getUsers())) {
      op->replaceUsesOfWith(rhsArgs, rhsNDWireOp.getResult());
    }
  }

  // Get lhs and rhs EndOp, a handshake.func can only have a single EndOp, TODO
  // check
  EndOp lhsEndOp = *lhsFuncOp.getOps<EndOp>().begin();
  EndOp rhsEndOp = *rhsFuncOp.getOps<EndOp>().begin();

  // Create comparison logic
  llvm::SmallVector<Value> eqResults;
  for (unsigned i = 0; i < lhsEndOp.getOperands().size(); ++i) {
    Value lhsResult = lhsEndOp.getOperand(i);
    Value rhsResult = rhsEndOp.getOperand(i);

    NDWireOp lhsNDWireOp = builder.create<NDWireOp>(forkOp.getLoc(), lhsResult);
    NDWireOp rhsNDWireOp = builder.create<NDWireOp>(forkOp.getLoc(), rhsResult);
    BufferOp lhsEndBufferOp = builder.create<BufferOp>(forkOp.getLoc(), lhsNDWireOp.getResult(), TimingInfo::oehb(), 3);
    BufferOp rhsEndBufferOp = builder.create<BufferOp>(forkOp.getLoc(), rhsNDWireOp.getResult(), TimingInfo::oehb(), 3);
    CmpIOp compOp = builder.create<CmpIOp>(builder.getUnknownLoc(), CmpIPredicate::eq, lhsEndBufferOp.getResult(),
                                           rhsEndBufferOp.getResult());
    eqResults.push_back(compOp.getResult());
  }

  EndOp newEndOp = builder.create<EndOp>(builder.getUnknownLoc(), eqResults);
  // TODO put in correct bb
  // newEndOp->setAttr(bb3);

  // Delete old end operation, we can only have one end operation in a function
  rhsEndOp.erase();
  lhsEndOp.erase();

  // Move operations from lhs to new
  Operation *previousOp = forkOp;
  for (Operation &op : llvm::make_early_inc_range(lhsFuncOp.getOps())) {
    op.moveAfter(previousOp);
    changeHandshakeBB(context, &op, 1);
    previousOp = &op;
  }

  // Move operations from rhs to new
  for (Operation &op : llvm::make_early_inc_range(rhsFuncOp.getOps())) {
    op.moveAfter(previousOp);
    changeHandshakeBB(context, &op, 2);
    previousOp = &op;
  }

  std::error_code EC;
  llvm::raw_fd_ostream fileStream("../tools/elastic-miter/out/comp/handshake_export.mlir", EC, llvm::sys::fs::OF_None);
  miterModule->print(fileStream, printingFlags);
  miterModule->print(llvm::outs(), printingFlags);

  if (!miterModule) {
    return failure();
  }
  return miterModule;
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerAsmPrinterCLOptions();

  // Register the supported dynamatic dialects and create a context
  DialectRegistry registry;
  dynamatic::registerAllDialects(registry);
  MLIRContext context(registry);

  exit(failed(createElasticMiter(context)));
}