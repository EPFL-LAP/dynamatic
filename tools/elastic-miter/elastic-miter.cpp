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

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"

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

static cl::opt<std::string> lhsFilename("lhs", cl::Prefix, cl::Required, cl::desc("<lhs MLIR input file>"), cl::cat(mainCategory));
static cl::opt<std::string> rhsFilename("rhs", cl::Prefix, cl::Required, cl::desc("<rhs MLIR input file>"), cl::cat(mainCategory));

static cl::opt<std::string> outputDir("o", cl::Prefix, cl::Required, cl::desc("<output directory>"), cl::cat(mainCategory));

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

static LogicalResult changeHandshakeName(OpBuilder &builder, Operation *op, std::string name) {
  StringAttr nameAttr = builder.getStringAttr(name);
  op->setAttr("handshake.name", nameAttr);
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

// TODO clean this up, documentation, do not pass newBlock, use FailureOr
static FuncOp buildNewFunc(OpBuilder builder, Block *newBlock, FuncOp &lhsFuncOp, FuncOp &rhsFuncOp) {
  // Copy the Result Names from lhs but prefix it with EQ_
  SmallVector<Attribute> prefixedResAttr;
  for (Attribute attr : lhsFuncOp.getResNames()) {
    auto strAttr = attr.dyn_cast<StringAttr>();
    if (strAttr) {
      prefixedResAttr.push_back(builder.getStringAttr("EQ_" + strAttr.getValue().str()));
    }
  }

  // Check equality of interfaces TODO should we really do this in this function
  if (lhsFuncOp.getArgumentTypes() != rhsFuncOp.getArgumentTypes()) {
    return nullptr;
  }
  if (lhsFuncOp.getResultTypes() != rhsFuncOp.getResultTypes()) {
    return nullptr;
  }

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
static FailureOr<OwningOpRef<ModuleOp>> createElasticMiter(MLIRContext &context, StringRef outputDir) {

  // TODO use arguments also check for existance
  OwningOpRef<ModuleOp> lhsModule = parseSourceFile<ModuleOp>(lhsFilename, &context);
  OwningOpRef<ModuleOp> rhsModule = parseSourceFile<ModuleOp>(rhsFilename, &context);

  // The module can only have one function so we just take the first element
  // TODO add a check for this
  FuncOp lhsFuncOp = *lhsModule->getOps<FuncOp>().begin();
  FuncOp rhsFuncOp = *rhsModule->getOps<FuncOp>().begin();

  OpBuilder builder(&context);

  OwningOpRef<ModuleOp> miterModule = ModuleOp::create(builder.getUnknownLoc());

  // Create a new function
  Block *newBlock = new Block();
  FuncOp newFuncOp = buildNewFunc(builder, newBlock, lhsFuncOp, rhsFuncOp);
  if (!newFuncOp)
    return failure();

  // Add the function to the module
  miterModule->push_back(newFuncOp);

  /* TODO:
  1: Name all auxillary operations
  */

  // Add "lhs_" to all operations in the LHS funcOp
  for (Operation &op : lhsFuncOp.getOps()) {
    if (failed(prefixOperation(op, "lhs_")))
      return failure();
  }
  // Add "rhs_" to all operations in the RHS funcOp
  for (Operation &op : rhsFuncOp.getOps()) {
    if (failed(prefixOperation(op, "rhs_")))
      return failure();
  }

  builder.setInsertionPointToStart(newBlock);

  llvm::json::Array inputBufferNames;
  llvm::json::Array outputBufferNames;
  llvm::json::Array ndwireNames;
  llvm::json::Array eqNames;

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

    std::string lhsBufName = "lhs_in_buf_" + lhsFuncOp.getArgName(i).str();
    std::string rhsBufName = "rhs_in_buf_" + lhsFuncOp.getArgName(i).str();
    std::string lhsNDwName = "lhs_in_ndw_" + lhsFuncOp.getArgName(i).str();
    std::string rhsNDwName = "rhs_in_ndw_" + lhsFuncOp.getArgName(i).str();
    forkOp = builder.create<ForkOp>(newFuncOp.getLoc(), miterArgs, 2);
    changeHandshakeName(builder, forkOp, "fork_" + lhsFuncOp.getArgName(i).str());
    lhsBufferOp = builder.create<BufferOp>(forkOp.getLoc(), forkOp.getResults()[0], TimingInfo::oehb(), 3);
    changeHandshakeName(builder, lhsBufferOp, lhsBufName);
    rhsBufferOp = builder.create<BufferOp>(forkOp.getLoc(), forkOp.getResults()[1], TimingInfo::oehb(), 3);
    changeHandshakeName(builder, rhsBufferOp, rhsBufName);
    lhsNDWireOp = builder.create<NDWireOp>(forkOp.getLoc(), lhsBufferOp.getResult());
    changeHandshakeName(builder, lhsNDWireOp, lhsNDwName);
    rhsNDWireOp = builder.create<NDWireOp>(forkOp.getLoc(), rhsBufferOp.getResult());
    changeHandshakeName(builder, rhsNDWireOp, rhsNDwName);

    llvm::json::Array inputBufferPair;
    inputBufferPair.push_back(lhsBufName);
    inputBufferPair.push_back(rhsBufName);
    inputBufferNames.push_back(std::move(inputBufferPair));

    ndwireNames.push_back(std::move(lhsNDwName));
    ndwireNames.push_back(std::move(rhsNDwName));

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

  size_t lhsEndOpCount = std::distance(lhsFuncOp.getOps<EndOp>().begin(), lhsFuncOp.getOps<EndOp>().end());
  size_t rhsEndOpCount = std::distance(lhsFuncOp.getOps<EndOp>().begin(), lhsFuncOp.getOps<EndOp>().end());

  assert(lhsEndOpCount == 1 && rhsEndOpCount == 1 && "A handshake::FuncOp can only have one EndOp.");

  // Get lhs and rhs EndOp, after checking there is only one EndOp each
  EndOp lhsEndOp = *lhsFuncOp.getOps<EndOp>().begin();
  EndOp rhsEndOp = *rhsFuncOp.getOps<EndOp>().begin();

  // Create comparison logic
  llvm::SmallVector<Value> eqResults;
  for (unsigned i = 0; i < lhsEndOp.getOperands().size(); ++i) {
    Value lhsResult = lhsEndOp.getOperand(i);
    Value rhsResult = rhsEndOp.getOperand(i);

    std::string lhsBufName = "lhs_out_buf_" + lhsFuncOp.getArgName(i).str();
    std::string rhsBufName = "rhs_out_buf_" + lhsFuncOp.getArgName(i).str();
    std::string lhsNDwName = "lhs_out_ndw_" + lhsFuncOp.getArgName(i).str();
    std::string rhsNDwName = "rhs_out_ndw_" + lhsFuncOp.getArgName(i).str();
    std::string eqName = "out_eq_" + lhsFuncOp.getResName(i).str();

    NDWireOp lhsEndNDWireOp = builder.create<NDWireOp>(rhsNDWireOp.getLoc(), lhsResult);
    changeHandshakeName(builder, lhsEndNDWireOp, lhsNDwName);
    NDWireOp rhsEndNDWireOp = builder.create<NDWireOp>(rhsNDWireOp.getLoc(), rhsResult);
    changeHandshakeName(builder, rhsEndNDWireOp, rhsNDwName);
    BufferOp lhsEndBufferOp = builder.create<BufferOp>(rhsNDWireOp.getLoc(), lhsEndNDWireOp.getResult(), TimingInfo::oehb(), 3);
    changeHandshakeName(builder, lhsEndBufferOp, lhsBufName);
    BufferOp rhsEndBufferOp = builder.create<BufferOp>(rhsNDWireOp.getLoc(), rhsEndNDWireOp.getResult(), TimingInfo::oehb(), 3);
    changeHandshakeName(builder, rhsEndBufferOp, rhsBufName);
    CmpIOp compOp =
        builder.create<CmpIOp>(builder.getUnknownLoc(), CmpIPredicate::eq, lhsEndBufferOp.getResult(), rhsEndBufferOp.getResult());
    changeHandshakeName(builder, compOp, eqName);

    llvm::json::Array outputBufferPair;
    outputBufferPair.push_back(lhsBufName);
    outputBufferPair.push_back(rhsBufName);
    outputBufferNames.push_back(std::move(outputBufferPair));

    ndwireNames.push_back(std::move(lhsNDwName));
    ndwireNames.push_back(std::move(rhsNDwName));

    eqNames.push_back(std::move(eqName));

    eqResults.push_back(compOp.getResult());
  }

  EndOp newEndOp = builder.create<EndOp>(builder.getUnknownLoc(), eqResults);
  changeHandshakeName(builder, newEndOp, "end");

  // TODO put in correct bb
  // newEndOp->setAttr(bb3);

  // Delete old end operation, we can only have one end operation in a function
  rhsEndOp.erase();
  lhsEndOp.erase();

  // Move operations from lhs to new
  Operation *previousOp = rhsNDWireOp;
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
  // TODO checks and balances
  SmallString<128> outMLIRFile;
  llvm::sys::path::append(outMLIRFile, outputDir, "handshake_export.mlir");
  llvm::raw_fd_ostream fileStream(outMLIRFile, EC, llvm::sys::fs::OF_None);
  miterModule->print(fileStream, printingFlags);
  miterModule->print(llvm::outs(), printingFlags);

  llvm::json::Object jsonObject;

  jsonObject["input_buffers"] = std::move(inputBufferNames);
  jsonObject["out_buffers"] = std::move(outputBufferNames);
  jsonObject["ndwires"] = std::move(ndwireNames);
  jsonObject["eq"] = std::move(eqNames);

  SmallString<128> outJsonFile;
  llvm::sys::path::append(outJsonFile, outputDir, "elastic-miter-config.json");
  llvm::raw_fd_ostream jsonFileStream(outJsonFile, EC, llvm::sys::fs::OF_None);

  // Try to print properly
  llvm::json::Value jsonValue(std::move(jsonObject));
  jsonFileStream << jsonValue << "\n";
  llvm::outs() << jsonValue << "\n";

  if (!miterModule) {
    return failure();
  }
  return miterModule;
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(argc, argv, "Placeholder TODO.");

  // Register the supported dynamatic dialects and create a context
  DialectRegistry registry;
  dynamatic::registerAllDialects(registry);
  MLIRContext context(registry);

  exit(failed(createElasticMiter(context, outputDir)));
}