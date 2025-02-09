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

static cl::opt<std::string>
    lhsFilenameArg("lhs", cl::Prefix, cl::Required,
                   cl::desc("Specify the left-hand side (LHS) input file"),
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

  llvm::outs() << outputDir << "\n";

  // Create the (potentially nested) output directory
  if (auto ec = llvm::sys::fs::create_directories(outputDir); ec.value() != 0) {
    llvm::errs() << "Failed to create output directory\n" << ec.message();
    return failure();
  }

  // Create the the handshake miter file
  SmallString<128> outMLIRFile;
  llvm::sys::path::append(outMLIRFile, outputDir, mlirFilename);
  llvm::raw_fd_ostream fileStream(outMLIRFile, ec, llvm::sys::fs::OF_None);

  // fileStream << "adhsadhibjkbud";

  mod->print(llvm::outs(), printingFlags);
  mod->print(fileStream, printingFlags);
  fileStream.close();

  if (!jsonObject.empty()) {
    // Create the JSON config file
    SmallString<128> outJsonFile;
    llvm::sys::path::append(outJsonFile, outputDir,
                            "elastic-miter-config.json");
    llvm::raw_fd_ostream jsonFileStream(outJsonFile, ec,
                                        llvm::sys::fs::OF_None);

    // Convert the JSON object to a JSON value in order to be printed
    llvm::json::Value jsonValue(std::move(jsonObject));
    // Print the JSON to the file with proper formatting
    jsonFileStream << llvm::formatv("{0:2}", jsonValue);
    jsonFileStream.close();
  }

  return success();
}

// This creates an elastic-miter module given the path to two MLIR files. The
// files need to contain exactely one module each. Each module needs to contain
// exactely one handshake.func.
static FailureOr<OwningOpRef<ModuleOp>>
createReachabilityCircuit(MLIRContext &context, StringRef filename) {

  OwningOpRef<ModuleOp> mod = parseSourceFile<ModuleOp>(filename, &context);
  if (!mod)
    return failure();

  // Get the LHS FuncOp from the LHS module, also check the
  auto funcOrFailure = getModuleFuncOpAndCheck(mod.get());
  if (failed(funcOrFailure))
    return failure();
  FuncOp funcOp = funcOrFailure.value();

  OpBuilder builder(&context);

  for (Block &block : funcOp.getBlocks()) {
    builder.setInsertionPointToStart(&block);
  }

  Operation *nextLocation;

  // Create the input side auxillary logic:
  // Every primary input of the LHS/RHS circuits is connected to a fork, which
  // splits the data for both sides under test. Both output sides of the fork
  // are connected to a buffer. The output of the buffer is connected to a
  // non-deterministic wire (ND wire), which models arbitrary backpressure
  // from surrounding circuits. The output of the ND wire is then connected to
  // the original input of the circuits. This completely decouples the
  // operation of the two circuits.
  for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
    BlockArgument args = funcOp.getArgument(i);

    std::string ndwName = "in_ndw_" + funcOp.getArgName(i).str();

    NDWireOp ndWireOp = builder.create<NDWireOp>(funcOp.getLoc(), args);
    setHandshakeAttributes(builder, ndWireOp, 0, ndwName);

    nextLocation = ndWireOp;

    // Use the newly created NDwire's output instead of the original argument in
    // the funcOp's operations
    for (Operation *op : llvm::make_early_inc_range(args.getUsers())) {
      if (op != ndWireOp)
        op->replaceUsesOfWith(args, ndWireOp.getResult());
    }
  }

  size_t endOpCount = std::distance(funcOp.getOps<EndOp>().begin(),
                                    funcOp.getOps<EndOp>().end());

  if (endOpCount != 1) {
    llvm::errs() << "The provided FuncOp is invalid. It needs to contain "
                    "exactly one EndOp.\n";
    return failure();
  }

  // GetEndOp, after checking there is only one EndOp
  EndOp endOp = *funcOp.getOps<EndOp>().begin();

  // Create the output side auxillary logic:
  // Every output of the LHS/RHS circuits is connected to a non-derministic
  // wire (ND wire). The output of the ND wire is connected to the respective
  // primary output.
  llvm::SmallVector<Value> eqResults;
  for (unsigned i = 0; i < endOp.getOperands().size(); ++i) {
    Value result = endOp.getOperand(i);

    std::string ndwName = "out_ndw_" + funcOp.getResName(i).str();

    NDWireOp endNDWireOp = builder.create<NDWireOp>(endOp->getLoc(), result);
    setHandshakeAttributes(builder, endNDWireOp, 3, ndwName);

    // Use the newly created NDwire's output instead of the original argument in
    // the FuncOp's operations
    for (Operation *op : llvm::make_early_inc_range(result.getUsers())) {
      if (op != endNDWireOp)
        op->replaceUsesOfWith(result, endNDWireOp.getResult());
    }
  }

  return mod;
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

  auto ret = createReachabilityCircuit(context, lhsFilenameArg);
  if (failed(ret)) {
    llvm::errs() << "Failed to create elastic-miter module.\n";
    return 1;
  }
  auto mod = ret.value().get();

  std::string mlirFilename =
      "ndwirererd_" +
      std::filesystem::path(lhsFilenameArg.getValue()).stem().string() +
      ".mlir";

  exit(failed(createFiles(outputDir, mlirFilename, mod, llvm::json::Object())));
}