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

//     _____ _      _____    _____      _
//    / ____| |    |_   _|  / ____|    | |
//   | |    | |      | |   | (___   ___| |_ _   _ _ __
//   | |    | |      | |    \___ \ / _ \ __| | | | '_ \
//   | |____| |____ _| |_   ____) |  __/ |_| |_| | |_) |
//    \_____|______|_____| |_____/ \___|\__|\__,_| .__/
//                                               | |
//                                               |_|

static cl::OptionCategory mainCategory("elastic-miter Options");

static cl::opt<std::string>
    lhsFilenameArg("lhs", cl::Prefix, cl::Required,
                   cl::desc("Specify the left-hand side (LHS) input file"),
                   cl::cat(mainCategory));
static cl::opt<std::string>
    rhsFilenameArg("rhs", cl::Prefix, cl::Required,
                   cl::desc("Specify the right-hand side (RHS) input file"),
                   cl::cat(mainCategory));

static cl::opt<std::string> outputDir("o", cl::Prefix, cl::Required,
                                      cl::desc("Specify output directory"),
                                      cl::cat(mainCategory));

//    ______ _           _   _             __  __ _ _
//   |  ____| |         | | (_)           |  \/  (_) |
//   | |__  | | __ _ ___| |_ _  ___ ______| \  / |_| |_ ___ _ __
//   |  __| | |/ _` / __| __| |/ __|______| |\/| | | __/ _ \ '__|
//   | |____| | (_| \__ \ |_| | (__       | |  | | | ||  __/ |
//   |______|_|\__,_|___/\__|_|\___|      |_|  |_|_|\__\___|_|
//
//

static void setHandshakeBB(OpBuilder &builder, Operation *op, int bb) {
  auto ui32 = IntegerType::get(builder.getContext(), 32,
                               IntegerType::SignednessSemantics::Unsigned);
  auto attr = IntegerAttr::get(ui32, bb);
  op->setAttr("handshake.bb", attr);
}

static void setHandshakeName(OpBuilder &builder, Operation *op,
                             const std::string &name) {
  StringAttr nameAttr = builder.getStringAttr(name);
  op->setAttr("handshake.name", nameAttr);
}

static void setHandshakeAttributes(OpBuilder &builder, Operation *op, int bb,
                                   const std::string &name) {
  setHandshakeBB(builder, op, bb);
  setHandshakeName(builder, op, name);
}

// Add a prefix to the handshake.name attribute of an operation.
// This avoids naming conflicts when merging two functions together.
// Returns failure() when the operation doesn't already have a handshake.name
static LogicalResult prefixOperation(Operation &op, const std::string &prefix) {
  StringAttr nameAttr = op.getAttrOfType<StringAttr>("handshake.name");
  if (!nameAttr)
    return failure();

  std::string newName = prefix + nameAttr.getValue().str();

  op.setAttr("handshake.name", StringAttr::get(op.getContext(), newName));

  return success();
}

static FailureOr<std::pair<FuncOp, Block *>>
buildNewFuncWithBlock(OpBuilder builder, const std::string &name,
                      ArrayRef<Type> inputTypes, ArrayRef<Type> outputTypes,
                      NamedAttribute argNamedAttr,
                      NamedAttribute resNamedAttr) {

  ArrayRef<NamedAttribute> funcAttr({argNamedAttr, resNamedAttr});

  FunctionType funcType = builder.getFunctionType(inputTypes, outputTypes);

  // Create the new function
  FuncOp newFuncOp =
      builder.create<FuncOp>(builder.getUnknownLoc(), name, funcType, funcAttr);

  // Add an entry block to the function
  Block *newEntryBlock = newFuncOp.addEntryBlock();

  return std::make_pair(newFuncOp, newEntryBlock);
}

// Build a elastic-miter template function with the interface of the LHS FuncOp.
// The output interface names have EQ_ prefixed.
static FailureOr<std::pair<FuncOp, Block *>>
buildEmptyMiterFuncOp(OpBuilder builder, FuncOp &lhsFuncOp, FuncOp &rhsFuncOp) {

  // Check equality of interfaces
  if (lhsFuncOp.getArgumentTypes() != rhsFuncOp.getArgumentTypes()) {
    llvm::errs() << "The input interfaces are not equivalent\n";
    return failure();
  }
  if (lhsFuncOp.getResultTypes() != rhsFuncOp.getResultTypes()) {
    llvm::errs() << "The output interfaces are not equivalent\n";
    return failure();
  }

  // Create the resNames attribute of the new miter FuncOp. The names are based
  // on the LHS resNames but are prefixed with EQ_
  SmallVector<Attribute> prefixedResAttr;
  for (Attribute attr : lhsFuncOp.getResNames()) {
    auto strAttr = attr.dyn_cast<StringAttr>();
    if (strAttr) {
      prefixedResAttr.push_back(
          builder.getStringAttr("EQ_" + strAttr.getValue().str()));
    }
  }

  // Create the argNames namedAttribute for the new miter funcOp, this is just a
  // copy of the LHS argNames
  auto argNamedAttr = builder.getNamedAttr("argNames", lhsFuncOp.getArgNames());
  // Create the resNames namedAttribute for the new miter funcOp, this is just a
  // copy of the LHS resNames with an added EQ_ prefix
  auto resNamedAttr =
      builder.getNamedAttr("resNames", builder.getArrayAttr(prefixedResAttr));

  size_t nrOfResults = lhsFuncOp.getResultTypes().size();
  ChannelType i1ChannelType = ChannelType::get(builder.getI1Type());

  // Create a vector with to replace all the output channels with type
  // !handshake.channel<i1>
  llvm::SmallVector<Type> outputTypes(nrOfResults, i1ChannelType);

  // Create the elastic-miter function
  return buildNewFuncWithBlock(builder, "elastic-miter",
                               lhsFuncOp.getArgumentTypes(), outputTypes,
                               argNamedAttr, resNamedAttr);
}

static LogicalResult createFiles(StringRef outputDir, ModuleOp mod,
                                 llvm::json::Object jsonObject) {

  std::error_code ec;
  OpPrintingFlags printingFlags;

  // Create the (potentially nested) output directory
  if (auto ec = llvm::sys::fs::create_directories(outputDir); ec.value() != 0) {
    llvm::errs() << "Failed to create output directory\n" << ec.message();
    return failure();
  }

  // Create the the handshake miter file
  SmallString<128> outMLIRFile;
  llvm::sys::path::append(outMLIRFile, outputDir, "handshake_miter.mlir");
  llvm::raw_fd_ostream fileStream(outMLIRFile, ec, llvm::sys::fs::OF_None);

  mod->print(fileStream, printingFlags);
  fileStream.close();

  // Create the JSON config file
  SmallString<128> outJsonFile;
  llvm::sys::path::append(outJsonFile, outputDir, "elastic-miter-config.json");
  llvm::raw_fd_ostream jsonFileStream(outJsonFile, ec, llvm::sys::fs::OF_None);

  // Convert the JSON object to a JSON value in order to be printed
  llvm::json::Value jsonValue(std::move(jsonObject));
  // Print the JSON to the file with proper formatting
  jsonFileStream << llvm::formatv("{0:2}", jsonValue);
  jsonFileStream.close();

  return success();
}

// This creates an elastic-miter module given the path to two MLIR files. The
// files need to contain exactely one module each. Each module needs to contain
// exactely one handshake.func.
static FailureOr<std::pair<ModuleOp, llvm::json::Object>>
createElasticMiter(MLIRContext &context, StringRef lhsFilename,
                   StringRef rhsFilename) {

  OwningOpRef<ModuleOp> lhsModule =
      parseSourceFile<ModuleOp>(lhsFilename, &context);
  if (!lhsModule)
    return failure();
  OwningOpRef<ModuleOp> rhsModule =
      parseSourceFile<ModuleOp>(rhsFilename, &context);
  if (!rhsModule)
    return failure();

  size_t lhsFuncOpCount = std::distance(lhsModule->getOps<FuncOp>().begin(),
                                        lhsModule->getOps<FuncOp>().end());
  size_t rhsFuncOpCount = std::distance(rhsModule->getOps<FuncOp>().begin(),
                                        rhsModule->getOps<FuncOp>().end());

  if (lhsFuncOpCount != 1 && rhsFuncOpCount != 1) {
    llvm::errs() << "The provided module is invalid. It needs to contain "
                    "exactely one FuncOp.\n";
    return failure();
  }

  // We know the module contains one FuncOp. So we can take take first element.
  FuncOp lhsFuncOp = *lhsModule->getOps<FuncOp>().begin();
  FuncOp rhsFuncOp = *rhsModule->getOps<FuncOp>().begin();

  OpBuilder builder(&context);

  ModuleOp miterModule = ModuleOp::create(builder.getUnknownLoc());
  if (!miterModule) {
    return failure();
  }

  // Create a new FuncOp containing an entry Block
  auto ret = buildEmptyMiterFuncOp(builder, lhsFuncOp, rhsFuncOp);
  if (failed(ret))
    return failure();

  // Unpack the returned FuncOp and entry Block
  auto [newFuncOp, newBlock] = ret.value();

  // Add the function to the module
  miterModule.push_back(newFuncOp);

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

  Operation *nextLocation;
  for (unsigned i = 0; i < lhsFuncOp.getNumArguments(); ++i) {
    BlockArgument lhsArgs = lhsFuncOp.getArgument(i);
    BlockArgument rhsArgs = rhsFuncOp.getArgument(i);
    BlockArgument miterArgs = newFuncOp.getArgument(i);

    std::string forkName = "in_fork_" + lhsFuncOp.getArgName(i).str();
    std::string lhsBufName = "lhs_in_buf_" + lhsFuncOp.getArgName(i).str();
    std::string rhsBufName = "rhs_in_buf_" + lhsFuncOp.getArgName(i).str();
    std::string lhsNdwName = "lhs_in_ndw_" + lhsFuncOp.getArgName(i).str();
    std::string rhsNdwName = "rhs_in_ndw_" + lhsFuncOp.getArgName(i).str();

    ForkOp forkOp = builder.create<ForkOp>(newFuncOp.getLoc(), miterArgs, 2);
    setHandshakeAttributes(builder, forkOp, 0, forkName);

    BufferOp lhsBufferOp = builder.create<BufferOp>(
        forkOp.getLoc(), forkOp.getResults()[0], TimingInfo::oehb(), 3);
    BufferOp rhsBufferOp = builder.create<BufferOp>(
        forkOp.getLoc(), forkOp.getResults()[1], TimingInfo::oehb(), 3);
    setHandshakeAttributes(builder, lhsBufferOp, 0, lhsBufName);
    setHandshakeAttributes(builder, rhsBufferOp, 0, rhsBufName);

    NDWireOp lhsNDWireOp =
        builder.create<NDWireOp>(forkOp.getLoc(), lhsBufferOp.getResult());
    NDWireOp rhsNDWireOp =
        builder.create<NDWireOp>(forkOp.getLoc(), rhsBufferOp.getResult());
    setHandshakeAttributes(builder, lhsNDWireOp, 0, lhsNdwName);
    setHandshakeAttributes(builder, rhsNDWireOp, 0, rhsNdwName);

    nextLocation = rhsNDWireOp;

    llvm::json::Array inputBufferPair;
    inputBufferPair.push_back(lhsBufName);
    inputBufferPair.push_back(rhsBufName);
    inputBufferNames.push_back(std::move(inputBufferPair));

    ndwireNames.push_back(std::move(lhsNdwName));
    ndwireNames.push_back(std::move(rhsNdwName));

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

  size_t lhsEndOpCount = std::distance(lhsFuncOp.getOps<EndOp>().begin(),
                                       lhsFuncOp.getOps<EndOp>().end());
  size_t rhsEndOpCount = std::distance(rhsFuncOp.getOps<EndOp>().begin(),
                                       rhsFuncOp.getOps<EndOp>().end());

  if (lhsEndOpCount != 1 && rhsEndOpCount != 1) {
    llvm::errs() << "The provided FuncOp is invalid. It needs to contain "
                    "exactely one EndOp.\n";
    return failure();
  }

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

    NDWireOp lhsEndNDWireOp;
    NDWireOp rhsEndNDWireOp;

    lhsEndNDWireOp =
        builder.create<NDWireOp>(nextLocation->getLoc(), lhsResult);
    rhsEndNDWireOp =
        builder.create<NDWireOp>(nextLocation->getLoc(), rhsResult);

    setHandshakeAttributes(builder, lhsEndNDWireOp, 3, lhsNDwName);
    setHandshakeAttributes(builder, rhsEndNDWireOp, 3, rhsNDwName);

    BufferOp lhsEndBufferOp = builder.create<BufferOp>(
        nextLocation->getLoc(), lhsEndNDWireOp.getResult(), TimingInfo::oehb(),
        3);
    BufferOp rhsEndBufferOp = builder.create<BufferOp>(
        nextLocation->getLoc(), rhsEndNDWireOp.getResult(), TimingInfo::oehb(),
        3);
    setHandshakeAttributes(builder, lhsEndBufferOp, 3, lhsBufName);
    setHandshakeAttributes(builder, rhsEndBufferOp, 3, rhsBufName);

    CmpIOp compOp = builder.create<CmpIOp>(
        builder.getUnknownLoc(), CmpIPredicate::eq, lhsEndBufferOp.getResult(),
        rhsEndBufferOp.getResult());
    setHandshakeAttributes(builder, compOp, 3, eqName);

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
  setHandshakeAttributes(builder, newEndOp, 3, "end");

  // Delete old end operation, we can only have one end operation in a function
  rhsEndOp.erase();
  lhsEndOp.erase();

  // Move operations from lhs to the new miter FuncOp and set the handshake.bb
  Operation *previousOp = nextLocation;
  for (Operation &op : llvm::make_early_inc_range(lhsFuncOp.getOps())) {
    op.moveAfter(previousOp);
    setHandshakeBB(builder, &op, 1);
    previousOp = &op;
  }

  // Move operations from rhs to the new miter FuncOp and set the handshake.bb
  for (Operation &op : llvm::make_early_inc_range(rhsFuncOp.getOps())) {
    op.moveAfter(previousOp);
    setHandshakeBB(builder, &op, 2);
    previousOp = &op;
  }

  llvm::json::Object jsonObject;

  jsonObject["input_buffers"] = std::move(inputBufferNames);
  jsonObject["out_buffers"] = std::move(outputBufferNames);
  jsonObject["ndwires"] = std::move(ndwireNames);
  jsonObject["eq"] = std::move(eqNames);

  return std::make_pair(miterModule, jsonObject);
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(
      argc, argv,
      "Creates an elastic-miter module in the handshake dialect.\n"
      "Takes two MLIR files as input. The files need to contain exactely one "
      "module each.\nEach module needs to contain exactely one handshake.func. "
      "\nThe resulting miter MLIR file and JSON config file are placed in the "
      "specified output directory.");

  // Register the supported dynamatic dialects and create a context
  DialectRegistry registry;
  dynamatic::registerAllDialects(registry);
  MLIRContext context(registry);

  auto ret = createElasticMiter(context, lhsFilenameArg, rhsFilenameArg);
  if (failed(ret)) {
    llvm::errs() << "Failed to create elastic-miter module.\n";
    return 1;
  }
  auto [miterModule, json] = ret.value();

  exit(failed(createFiles(outputDir, miterModule, json)));
}