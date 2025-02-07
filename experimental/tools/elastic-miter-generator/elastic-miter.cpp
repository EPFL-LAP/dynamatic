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

#include <filesystem>

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

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/InitAllDialects.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"

namespace cl = llvm::cl;
using namespace mlir;
using namespace dynamatic::handshake;

// CLI Settings

static cl::OptionCategory mainCategory("elastic-miter Options");

static cl::opt<size_t>
    nrOfBufferSlots("bufferSlots", cl::Prefix, cl::Required,
                    cl::desc("Specify the number of required buffers."),
                    cl::cat(mainCategory));
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

static void setHandshakeName(OpBuilder &builder, Operation *op,
                             const std::string &name) {
  StringAttr nameAttr = builder.getStringAttr(name);
  op->setAttr(dynamatic::NameAnalysis::ATTR_NAME, nameAttr);
}

static void setHandshakeAttributes(OpBuilder &builder, Operation *op, int bb,
                                   const std::string &name) {
  dynamatic::setBB(op, bb);
  setHandshakeName(builder, op, name);
}

// Add a prefix to the handshake.name attribute of an operation.
// This avoids naming conflicts when merging two functions together.
// Returns failure() when the operation doesn't already have a handshake.name
static LogicalResult prefixOperation(Operation &op, const std::string &prefix) {
  StringAttr nameAttr =
      op.getAttrOfType<StringAttr>(dynamatic::NameAnalysis::ATTR_NAME);
  if (!nameAttr)
    return failure();

  std::string newName = prefix + nameAttr.getValue().str();

  op.setAttr(dynamatic::NameAnalysis::ATTR_NAME,
             StringAttr::get(op.getContext(), newName));

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
// The result names have EQ_ prefixed.
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

  // The inputs to the elastic-miter circuit are the same as the input signature
  // of the LHS/RHS circuit. We assign them the same argument names as the LHS.
  auto argNamedAttr = builder.getNamedAttr("argNames", lhsFuncOp.getArgNames());
  // Create the resNames namedAttribute for the new miter funcOp, this is just a
  // copy of the LHS resNames with an added EQ_ prefix
  auto resNamedAttr =
      builder.getNamedAttr("resNames", builder.getArrayAttr(prefixedResAttr));

  size_t nrOfResults = lhsFuncOp.getResultTypes().size();
  ChannelType i1ChannelType = ChannelType::get(builder.getI1Type());

  // The outputs of the elastic miter are an EQ gate with a Boolean result
  // for every primary output of the original circuits.
  // Create a vector with to replace all the output channels with type
  // !handshake.channel<i1>
  llvm::SmallVector<Type> outputTypes(nrOfResults, i1ChannelType);

  // The name of the new function is: elastic_miter_<LHS_NAME>_<RHS_NAME>
  std::string miterName = "elastic_miter_" + lhsFuncOp.getNameAttr().str() +
                          "_" + rhsFuncOp.getNameAttr().str();

  // Create the elastic-miter function
  return buildNewFuncWithBlock(builder, miterName, lhsFuncOp.getArgumentTypes(),
                               outputTypes, argNamedAttr, resNamedAttr);
}

// Get the first and only non-external FuncOp from the module.
// Additionally check some properites:
// 1. The FuncOp is materialized (each Value is only used once).
// 2. There are no memory interfaces
// 3. Arguments  and results are all handshake.channel or handshake.control type
static FailureOr<FuncOp> getModuleFuncOpAndCheck(ModuleOp module) {
  // We only support one function per module
  FuncOp funcOp = nullptr;
  for (auto op : module.getOps<FuncOp>()) {
    if (op.isExternal())
      continue;
    if (funcOp) {
      llvm::errs() << "We currently only support one non-external "
                      "handshake function per module\n";
      return failure();
    }
    funcOp = op;
  }

  if (funcOp) {
    // Check that the FuncOp is materialized
    if (failed(dynamatic::verifyIRMaterialized(funcOp))) {
      llvm::errs() << dynamatic::ERR_NON_MATERIALIZED_FUNC;
      return failure();
    }
  }

  // Check that arguments are all handshake.channel or handshake.control type
  for (Type ty : funcOp.getArgumentTypes()) {
    if (!ty.isa<ChannelType, ControlType>()) {
      llvm::errs() << "All arguments need to be of handshake.channel or "
                      "handshake.control type\n";
      return failure();
    }
  }

  // Check that results are all handshake.channel or handshake.control type
  for (Type ty : funcOp.getResultTypes()) {
    if (!ty.isa<ChannelType, ControlType>()) {
      llvm::errs() << "All results need to be of handshake.channel or "
                      "handshake.control type\n";
      return failure();
    }
  }

  // Check that there are no memory interfaces
  if (!funcOp.getOps<MemoryOpInterface>().empty()) {
    llvm::errs()
        << "There can be no Memory Interfaces as they are not elastic.";
    return failure();
  }
  return funcOp;
}

static LogicalResult createFiles(StringRef outputDir, StringRef mlirFilename,
                                 ModuleOp mod, llvm::json::Object jsonObject) {

  std::error_code ec;
  OpPrintingFlags printingFlags;

  // Create the (potentially nested) output directory
  if (auto ec = llvm::sys::fs::create_directories(outputDir); ec.value() != 0) {
    llvm::errs() << "Failed to create output directory\n" << ec.message();
    return failure();
  }

  // Create the the handshake miter file
  SmallString<128> outMLIRFile;
  llvm::sys::path::append(outMLIRFile, outputDir, mlirFilename);
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
                   StringRef rhsFilename, size_t bufferSlots) {

  OwningOpRef<ModuleOp> lhsModule =
      parseSourceFile<ModuleOp>(lhsFilename, &context);
  if (!lhsModule)
    return failure();
  OwningOpRef<ModuleOp> rhsModule =
      parseSourceFile<ModuleOp>(rhsFilename, &context);
  if (!rhsModule)
    return failure();

  // Get the LHS FuncOp from the LHS module, also check the
  auto funcOrFailure = getModuleFuncOpAndCheck(lhsModule.get());
  if (failed(funcOrFailure))
    return failure();
  FuncOp lhsFuncOp = funcOrFailure.value();

  // Get the RHS FuncOp from the RHS module
  funcOrFailure = getModuleFuncOpAndCheck(rhsModule.get());
  if (failed(funcOrFailure))
    return failure();
  FuncOp rhsFuncOp = funcOrFailure.value();

  OpBuilder builder(&context);

  ModuleOp miterModule = ModuleOp::create(builder.getUnknownLoc());
  if (!miterModule) {
    return failure();
  }

  // Create a new FuncOp containing an entry Block
  auto pairOrFailure = buildEmptyMiterFuncOp(builder, lhsFuncOp, rhsFuncOp);
  if (failed(pairOrFailure))
    return failure();

  // Unpack the returned FuncOp and entry Block
  auto [newFuncOp, newBlock] = pairOrFailure.value();

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

  // We need to keep track of the names to add them to the JSON config file
  llvm::json::Array inputBufferNames;
  llvm::json::Array outputBufferNames;
  llvm::json::Array ndwireNames;
  llvm::json::Array eqNames;

  Operation *nextLocation;

  // Create the input side auxillary logic:
  // Every primary input of the LHS/RHS circuits is connected to a fork, which
  // splits the data for both sides under test. Both output sides of the fork
  // are connected to a buffer. The output of the buffer is connected to a
  // non-deterministic wire (ND wire), which models arbitrary backpressure
  // from surrounding circuits. The output of the ND wire is then connected to
  // the original input of the circuits. This completely decouples the
  // operation of the two circuits.
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

    BufferOp lhsBufferOp =
        builder.create<BufferOp>(forkOp.getLoc(), forkOp.getResults()[0],
                                 TimingInfo::oehb(), bufferSlots);
    BufferOp rhsBufferOp =
        builder.create<BufferOp>(forkOp.getLoc(), forkOp.getResults()[1],
                                 TimingInfo::oehb(), bufferSlots);
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

    // Use the newly created fork's output instead of the original argument in
    // the lhsFuncOp's operations
    for (Operation *op : llvm::make_early_inc_range(lhsArgs.getUsers()))
      op->replaceUsesOfWith(lhsArgs, lhsNDWireOp.getResult());

    // Use the newly created fork's output instead of the original argument in
    // the rhsFuncOp's operations
    for (Operation *op : llvm::make_early_inc_range(rhsArgs.getUsers()))
      op->replaceUsesOfWith(rhsArgs, rhsNDWireOp.getResult());
  }

  size_t lhsEndOpCount = std::distance(lhsFuncOp.getOps<EndOp>().begin(),
                                       lhsFuncOp.getOps<EndOp>().end());
  size_t rhsEndOpCount = std::distance(rhsFuncOp.getOps<EndOp>().begin(),
                                       rhsFuncOp.getOps<EndOp>().end());

  if (lhsEndOpCount != 1 && rhsEndOpCount != 1) {
    llvm::errs() << "The provided FuncOp is invalid. It needs to contain "
                    "exactly one EndOp.\n";
    return failure();
  }

  // Get lhs and rhs EndOp, after checking there is only one EndOp each
  EndOp lhsEndOp = *lhsFuncOp.getOps<EndOp>().begin();
  EndOp rhsEndOp = *rhsFuncOp.getOps<EndOp>().begin();

  // Create the output side auxillary logic:
  // Every output of the LHS/RHS circuits are connected to a non-derministic
  // wire (ND wire), which models arbitrary backpressure from surrounding
  // circuits. The output of the ND wire is connected to a buffer. This
  // completely decouples the operation of the two circuits. The buffers then
  // are connected pairwise to a comparator to check that the outputs are
  // equivalent.
  llvm::SmallVector<Value> eqResults;
  for (unsigned i = 0; i < lhsEndOp.getOperands().size(); ++i) {
    Value lhsResult = lhsEndOp.getOperand(i);
    Value rhsResult = rhsEndOp.getOperand(i);

    std::string lhsBufName = "lhs_out_buf_" + lhsFuncOp.getResName(i).str();
    std::string rhsBufName = "rhs_out_buf_" + lhsFuncOp.getResName(i).str();
    std::string lhsNDwName = "lhs_out_ndw_" + lhsFuncOp.getResName(i).str();
    std::string rhsNDwName = "rhs_out_ndw_" + lhsFuncOp.getResName(i).str();
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
        bufferSlots);
    BufferOp rhsEndBufferOp = builder.create<BufferOp>(
        nextLocation->getLoc(), rhsEndNDWireOp.getResult(), TimingInfo::oehb(),
        bufferSlots);
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

  // Delete old end operation, we can only have one end operation in a
  // function
  rhsEndOp.erase();
  lhsEndOp.erase();

  // Move operations from lhs to the new miter FuncOp and set the handshake.bb
  Operation *previousOp = nextLocation;
  for (Operation &op : llvm::make_early_inc_range(lhsFuncOp.getOps())) {
    op.moveAfter(previousOp);
    dynamatic::setBB(&op, 1);
    previousOp = &op;
  }

  // Move operations from rhs to the new miter FuncOp and set the handshake.bb
  for (Operation &op : llvm::make_early_inc_range(rhsFuncOp.getOps())) {
    op.moveAfter(previousOp);
    dynamatic::setBB(&op, 2);
    previousOp = &op;
  }

  // Create a new jsonObject and put all required names into it
  llvm::json::Object jsonObject = *new llvm::json::Object;
  jsonObject["input_buffers"] = std::move(inputBufferNames);
  jsonObject["output_buffers"] = std::move(outputBufferNames);
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
      "module each.\nEach module needs to contain exactely one "
      "handshake.func. "
      "\nThe resulting miter MLIR file and JSON config file are placed in "
      "the "
      "specified output directory.");

  // Register the supported dynamatic dialects and create a context
  DialectRegistry registry;
  dynamatic::registerAllDialects(registry);
  MLIRContext context(registry);

  auto ret = createElasticMiter(context, lhsFilenameArg, rhsFilenameArg,
                                nrOfBufferSlots);
  if (failed(ret)) {
    llvm::errs() << "Failed to create elastic-miter module.\n";
    return 1;
  }
  auto [miterModule, json] = ret.value();

  std::string mlirFilename =
      "elastic_miter_" +
      std::filesystem::path(lhsFilenameArg.getValue()).stem().string() + "_" +
      std::filesystem::path(rhsFilenameArg.getValue()).stem().string() +
      ".mlir";

  exit(failed(createFiles(outputDir, mlirFilename, miterModule, json)));
}