//===- FabricGeneration.cpp ------------------------------------ *- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <filesystem>
#include <iterator>
#include <string>
#include <utility>

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"

#include "FabricGeneration.h"

using namespace mlir;
using namespace dynamatic::handshake;

namespace dynamatic::experimental {

void setHandshakeName(OpBuilder &builder, Operation *op,
                      const std::string &name) {
  StringAttr nameAttr = builder.getStringAttr(name);
  op->setAttr(dynamatic::NameAnalysis::ATTR_NAME, nameAttr);
}

std::string getHandshakeName(Operation *op) {
  StringAttr nameAttr =
      op->getAttrOfType<StringAttr>(dynamatic::NameAnalysis::ATTR_NAME);
  if (nameAttr)
    return nameAttr.getValue().str();
  return "";
}

void setHandshakeAttributes(OpBuilder &builder, Operation *op, int bb,
                            const std::string &name) {
  dynamatic::setBB(op, bb);
  setHandshakeName(builder, op, name);
}

// Add a prefix to the handshake.name attribute of an operation.
// This avoids naming conflicts when merging two functions together.
// Returns failure() when the operation doesn't already have a handshake.name
LogicalResult prefixOperation(Operation &op, const std::string &prefix) {
  StringAttr nameAttr =
      op.getAttrOfType<StringAttr>(dynamatic::NameAnalysis::ATTR_NAME);
  if (!nameAttr)
    return failure();

  std::string newName = prefix + nameAttr.getValue().str();

  op.setAttr(dynamatic::NameAnalysis::ATTR_NAME,
             StringAttr::get(op.getContext(), newName));

  return success();
}

std::pair<FuncOp, Block *> buildNewFuncWithBlock(llvm::StringRef name,
                                                 ArrayRef<Type> inputTypes,
                                                 ArrayRef<Type> outputTypes,
                                                 ArrayAttr argNames,
                                                 ArrayAttr resNames) {

  OpBuilder builder(argNames.getContext());

  auto argNamedAttr = builder.getNamedAttr("argNames", argNames);
  auto resNamedAttr = builder.getNamedAttr("resNames", resNames);

  SmallVector<NamedAttribute> funcAttr({argNamedAttr, resNamedAttr});

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
FailureOr<std::pair<FuncOp, Block *>>
buildEmptyMiterFuncOp(OpBuilder builder, FuncOp &lhsFuncOp, FuncOp &rhsFuncOp,
                      llvm::StringRef funcName) {

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
  auto argAttr = lhsFuncOp.getArgNames();
  // Create the resNames namedAttribute for the new miter funcOp, this is just a
  // copy of the LHS resNames with an added EQ_ prefix
  auto resAttr = builder.getArrayAttr(prefixedResAttr);

  ChannelType i1ChannelType = ChannelType::get(builder.getI1Type());

  // If the original output was of ChannelType, outputs of the elastic miter is
  // an EQ gate with a Boolean result. Otherwise it is the output of a join. The
  // ChannelTypes are replaced with !handshake.channel<i1>, independent of size,
  // ControlType are kept the same.
  llvm::SmallVector<Type> outputTypes;

  for (Type type : lhsFuncOp.getResultTypes()) {
    // If the type is a handshake::ControlType, keep it
    if (type.isa<handshake::ControlType>()) {
      outputTypes.push_back(type);
    } else {
      // Otherwise replace it with !handshake.channel<i1>
      outputTypes.push_back(i1ChannelType);
    }
  }

  // Now, you have the modified SmallVector<Type>
  mlir::ArrayRef<mlir::Type> newArrayRef(outputTypes);

  // Create the elastic-miter function
  return buildNewFuncWithBlock(funcName, lhsFuncOp.getArgumentTypes(),
                               outputTypes, argAttr, resAttr);
}

// Get the first and only non-external FuncOp from the module.
// Additionally check some properites:
// 1. The FuncOp is materialized (each Value is only used once).
// 2. There are no memory interfaces
// 3. Arguments  and results are all handshake.channel or handshake.control type
FailureOr<FuncOp> getModuleFuncOpAndCheck(ModuleOp module) {
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

// Prints the module to the provided file, creates the directory if it does not
// exist yet.
LogicalResult createMlirFile(const std::filesystem::path &mlirPath,
                             ModuleOp modul) {

  std::error_code ec;
  OpPrintingFlags printingFlags;

  // Create the output directory
  std::filesystem::create_directories(mlirPath.parent_path());

  // Create the the file at the path
  llvm::raw_fd_ostream fileStream(mlirPath.string(), ec,
                                  llvm::sys::fs::OF_None);

  modul->print(fileStream, printingFlags);
  fileStream.close();

  return success();
}

EndOp getUniqueEndOp(FuncOp funcOp) {
  auto endOps = funcOp.getOps<EndOp>();
  assert(std::distance(endOps.begin(), endOps.end()) == 1 &&
         "There should be exactly one EndOp in the function");
  return *endOps.begin();
}

// Create the reachability circuit by putting ND wires at all the in- and
// outputs
FailureOr<std::pair<ModuleOp, struct ElasticMiterConfig>>
createReachabilityCircuit(MLIRContext &context,
                          const std::filesystem::path &filename,
                          bool timingInsensitive) {

  OwningOpRef<ModuleOp> mod =
      parseSourceFile<ModuleOp>(filename.string(), &context);
  if (!mod)
    return failure();

  // Get the FuncOp from the module, also check some needed properties
  auto funcOrFailure = getModuleFuncOpAndCheck(mod.get());
  if (failed(funcOrFailure))
    return failure();
  FuncOp funcOp = funcOrFailure.value();

  OpBuilder builder(&context);

  for (Block &block : funcOp.getBlocks()) {
    builder.setInsertionPointToStart(&block);
  }

  // Create the config
  ElasticMiterConfig config;
  config.funcName = funcOp.getNameAttr().str();

  // Create the input side auxillary logic:
  // Every primary input of the LHS/RHS circuits is connected to a ND wire.
  // The output of the ND wire is then connected to the original input of the
  // circuit.
  for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
    BlockArgument arg = funcOp.getArgument(i);
    if (getHandshakeTypeBitWidth(arg.getType()) > 1)
      return funcOp.emitError("ElasticMiter currently supports only 1-bit or "
                              "control inputs.");

    if (!timingInsensitive) {
      std::string ndwName = "ndw_in_" + funcOp.getArgName(i).str();

      NDWireOp ndWireOp = builder.create<NDWireOp>(funcOp.getLoc(), arg);
      setHandshakeAttributes(builder, ndWireOp, 0, ndwName);

      // Use the newly created NDwire's output instead of the original argument
      // in the funcOp's operations
      for (Operation *op : llvm::make_early_inc_range(arg.getUsers())) {
        if (op != ndWireOp)
          op->replaceUsesOfWith(arg, ndWireOp.getResult());
      }
    }

    Attribute attr = funcOp.getArgNames()[i];
    auto strAttr = attr.dyn_cast<StringAttr>();
    config.arguments.push_back(
        std::make_pair(strAttr.getValue().str(), arg.getType()));
  }

  EndOp endOp = getUniqueEndOp(funcOp);

  // Create the output side auxillary logic:
  // Every output of the LHS/RHS circuits is connected to a non-derministic
  // wire (ND wire). The output of the ND wire is connected to the respective
  // primary output.
  for (unsigned i = 0; i < endOp.getOperands().size(); ++i) {
    Value result = endOp.getOperand(i);

    if (!timingInsensitive) {
      std::string ndwName = "ndw_out_" + funcOp.getResName(i).str();

      NDWireOp endNDWireOp = builder.create<NDWireOp>(endOp->getLoc(), result);
      setHandshakeAttributes(builder, endNDWireOp, 3, ndwName);

      // Use the newly created NDwire's output instead of the original argument
      // in the FuncOp's operations
      for (Operation *op : llvm::make_early_inc_range(result.getUsers())) {
        if (op != endNDWireOp)
          op->replaceUsesOfWith(result, endNDWireOp.getResult());
      }
    }

    Attribute attr = funcOp.getResNames()[i];
    auto strAttr = attr.dyn_cast<StringAttr>();
    config.results.push_back(
        std::make_pair(strAttr.getValue().str(), result.getType()));
  }

  return std::make_pair(mod.release(), config);
}

// This creates an elastic-miter module given the path to two MLIR files. The
// files need to contain exactely one module each. Each module needs to contain
// exactely one handshake.func.
FailureOr<std::pair<ModuleOp, struct ElasticMiterConfig>>
createElasticMiter(MLIRContext &context, ModuleOp lhsModule, ModuleOp rhsModule,
                   size_t bufferSlots, bool allowNonacceptance,
                   bool timingInsensitive) {

  // Get the LHS FuncOp from the LHS module, also check some required properties
  auto funcOrFailure = getModuleFuncOpAndCheck(lhsModule);
  if (failed(funcOrFailure)) {
    llvm::errs() << "Failed to get LHS FuncOp.\n";
    return failure();
  }
  FuncOp lhsFuncOp = funcOrFailure.value();

  // Get the RHS FuncOp from the RHS module
  funcOrFailure = getModuleFuncOpAndCheck(rhsModule);
  if (failed(funcOrFailure)) {
    llvm::errs() << "Failed to get RHS FuncOp.\n";
    return failure();
  }
  FuncOp rhsFuncOp = funcOrFailure.value();

  OpBuilder builder(&context);

  ModuleOp miterModule = ModuleOp::create(builder.getUnknownLoc());
  if (!miterModule) {
    llvm::errs() << "Failed to create new empty module.\n";
    return failure();
  }

  // Create the config
  ElasticMiterConfig config;
  config.lhsFuncName = lhsFuncOp.getNameAttr().str();
  config.rhsFuncName = rhsFuncOp.getNameAttr().str();

  // The name of the new function is: elastic_miter_<LHS_NAME>_<RHS_NAME>
  config.funcName =
      "elastic_miter_" + config.lhsFuncName + "_" + config.rhsFuncName;

  // Create a new FuncOp containing an entry Block
  auto pairOrFailure =
      buildEmptyMiterFuncOp(builder, lhsFuncOp, rhsFuncOp, config.funcName);
  if (failed(pairOrFailure)) {
    llvm::errs() << "Failed to create new funcOp.\n";
    return failure();
  }
  // Unpack the returned FuncOp and entry Block
  auto [newFuncOp, newBlock] = pairOrFailure.value();

  // Add the function to the module
  miterModule.push_back(newFuncOp);

  // Add "lhs_" to all operations in the LHS funcOp
  for (Operation &op : lhsFuncOp.getOps()) {
    if (failed(prefixOperation(op, "lhs_"))) {
      llvm::errs() << "Failed to prefix the LHS.\n";
      return failure();
    }
  }

  // Add "rhs_" to all operations in the RHS funcOp
  for (Operation &op : rhsFuncOp.getOps()) {
    if (failed(prefixOperation(op, "rhs_"))) {
      llvm::errs() << "Failed to prefix the RHS.\n";
      return failure();
    }
  }

  builder.setInsertionPointToStart(newBlock);

  // Create the input side auxillary logic:
  // Every primary input of the LHS/RHS circuits is connected to a fork, which
  // splits the data for both sides under test. Both output sides of the fork
  // are connected to a buffer. The output of the buffer is connected to a
  // non-deterministic wire (ND wire), which models arbitrary backpressure
  // from surrounding circuits. The output of the ND wire is then connected to
  // the original input of the circuits. This completely decouples the
  // operation of the two circuits.
  size_t inputBufferSlots = timingInsensitive ? 1 : bufferSlots;
  for (unsigned i = 0; i < lhsFuncOp.getNumArguments(); ++i) {
    Value lhsArg = lhsFuncOp.getArgument(i);
    Value rhsArg = rhsFuncOp.getArgument(i);
    Value miterArg = newFuncOp.getArgument(i);
    if (getHandshakeTypeBitWidth(miterArg.getType()) > 1) {
      return lhsFuncOp.emitError(
          "ElasticMiter currently supports only 1-bit or "
          "control inputs.");
    }

    std::string forkName = "in_fork_" + lhsFuncOp.getArgName(i).str();
    std::string lhsBufName = "lhs_in_buf_" + lhsFuncOp.getArgName(i).str();
    std::string rhsBufName = "rhs_in_buf_" + lhsFuncOp.getArgName(i).str();
    std::string lhsNdwName = "lhs_in_ndw_" + lhsFuncOp.getArgName(i).str();
    std::string rhsNdwName = "rhs_in_ndw_" + lhsFuncOp.getArgName(i).str();

    auto forkOp = builder.create<LazyForkOp>(newFuncOp.getLoc(), miterArg, 2);
    setHandshakeAttributes(builder, forkOp, BB_IN, forkName);
    config.inputForks.push_back(forkName);

    BufferOp lhsBufferOp = builder.create<BufferOp>(
        forkOp.getLoc(), forkOp.getResult()[0], inputBufferSlots,
        dynamatic::handshake::BufferType::FIFO_BREAK_DV);
    lhsBufferOp.setDebugCounter(true);
    setHandshakeAttributes(builder, lhsBufferOp, BB_IN, lhsBufName);
    Value lhsNDWireInput = lhsBufferOp.getResult();

    BufferOp rhsBufferOp = builder.create<BufferOp>(
        forkOp.getLoc(), forkOp.getResult()[1], inputBufferSlots,
        dynamatic::handshake::BufferType::FIFO_BREAK_DV);
    rhsBufferOp.setDebugCounter(true);
    setHandshakeAttributes(builder, rhsBufferOp, BB_IN, rhsBufName);
    Value rhsNDWireInput = rhsBufferOp.getResult();

    Value lhsInput = lhsNDWireInput;
    Value rhsInput = rhsNDWireInput;
    if (timingInsensitive) {
      lhsNdwName = "";
      rhsNdwName = "";
    } else {
      NDWireOp lhsNDWireOp =
          builder.create<NDWireOp>(forkOp.getLoc(), lhsNDWireInput);
      NDWireOp rhsNDWireOp =
          builder.create<NDWireOp>(forkOp.getLoc(), rhsNDWireInput);
      setHandshakeAttributes(builder, lhsNDWireOp, BB_IN, lhsNdwName);
      setHandshakeAttributes(builder, rhsNDWireOp, BB_IN, rhsNdwName);
      lhsInput = lhsNDWireOp.getResult();
      rhsInput = rhsNDWireOp.getResult();
    }

    config.inputBuffers.push_back(std::make_pair(lhsBufName, rhsBufName));
    config.inputNDWires.push_back(std::make_pair(lhsNdwName, rhsNdwName));

    // Use the newly created fork's output instead of the original argument in
    // the lhsFuncOp's operations
    for (Operation *op : llvm::make_early_inc_range(lhsArg.getUsers()))
      op->replaceUsesOfWith(lhsArg, lhsInput);

    // Use the newly created fork's output instead of the original argument in
    // the rhsFuncOp's operations
    for (Operation *op : llvm::make_early_inc_range(rhsArg.getUsers()))
      op->replaceUsesOfWith(rhsArg, rhsInput);

    Attribute attr = lhsFuncOp.getArgNames()[i];
    auto strAttr = attr.dyn_cast<StringAttr>();
    config.arguments.push_back(
        std::make_pair(strAttr.getValue().str(), lhsArg.getType()));
  }

  // Get lhs and rhs EndOp, after checking there is only one EndOp each
  EndOp lhsEndOp = getUniqueEndOp(lhsFuncOp);
  EndOp rhsEndOp = getUniqueEndOp(rhsFuncOp);

  // Create the output side auxillary logic:
  // Every output of the LHS/RHS circuits are connected to a non-derministic
  // wire (ND wire), which models arbitrary backpressure from surrounding
  // circuits. The output of the ND wire is connected to a buffer. This
  // completely decouples the operation of the two circuits. The buffers then
  // are connected pairwise to a comparator to check that the outputs are
  // equivalent.
  llvm::SmallVector<Value> miterResultValues;
  Location loc = newFuncOp.getLoc();
  for (unsigned i = 0; i < lhsEndOp.getOperands().size(); ++i) {
    Value lhsResult = lhsEndOp.getOperand(i);
    Value rhsResult = rhsEndOp.getOperand(i);

    std::string outName = lhsFuncOp.getResName(i).str();
    std::string lhsBufName = "lhs_out_buf_" + outName;
    std::string rhsBufName = "rhs_out_buf_" + outName;
    std::string lhsNDwName = "lhs_out_ndw_" + outName;
    std::string rhsNDwName = "rhs_out_ndw_" + outName;
    std::string lhsBlockerName = "lhs_out_bl_" + outName;
    std::string rhsBlockerName = "rhs_out_bl_" + outName;
    std::string eqName = "out_eq_" + outName;

    Value lhsBufferInput, rhsBufferInput;

    if (timingInsensitive) {
      lhsNDwName = "";
      rhsNDwName = "";

      lhsBufferInput = lhsResult;
      rhsBufferInput = rhsResult;
    } else {
      NDWireOp lhsEndNDWireOp;
      NDWireOp rhsEndNDWireOp;

      lhsEndNDWireOp = builder.create<NDWireOp>(loc, lhsResult);
      rhsEndNDWireOp = builder.create<NDWireOp>(loc, rhsResult);

      setHandshakeAttributes(builder, lhsEndNDWireOp, BB_OUT, lhsNDwName);
      setHandshakeAttributes(builder, rhsEndNDWireOp, BB_OUT, rhsNDwName);

      lhsBufferInput = lhsEndNDWireOp.getResult();
      rhsBufferInput = rhsEndNDWireOp.getResult();
    }

    Value lhsInput = lhsBufferInput;
    Value rhsInput = rhsBufferInput;
    if (timingInsensitive) {
      lhsBufName = "";
      rhsBufName = "";
    } else {
      BufferOp lhsEndBufferOp = builder.create<BufferOp>(
          loc, lhsBufferInput, bufferSlots,
          dynamatic::handshake::BufferType::FIFO_BREAK_DV);
      BufferOp rhsEndBufferOp = builder.create<BufferOp>(
          loc, rhsBufferInput, bufferSlots,
          dynamatic::handshake::BufferType::FIFO_BREAK_DV);

      setHandshakeAttributes(builder, lhsEndBufferOp, BB_OUT, lhsBufName);
      setHandshakeAttributes(builder, rhsEndBufferOp, BB_OUT, rhsBufName);

      lhsInput = lhsEndBufferOp.getResult();
      rhsInput = rhsEndBufferOp.getResult();
    }

    if (lhsResult.getType().isa<handshake::ControlType>()) {
      SmallVector<Value> joinInputs = {lhsInput, rhsInput};
      JoinOp joinOp =
          builder.create<JoinOp>(builder.getUnknownLoc(), joinInputs);
      setHandshakeAttributes(builder, joinOp, BB_OUT, eqName);
      miterResultValues.push_back(joinOp.getResult());
    } else {
      CmpIOp compOp = builder.create<CmpIOp>(
          builder.getUnknownLoc(), CmpIPredicate::eq, lhsInput, rhsInput);
      setHandshakeAttributes(builder, compOp, BB_OUT, eqName);
      miterResultValues.push_back(compOp.getResult());
    }

    config.outputBuffers.push_back(std::make_pair(lhsBufName, rhsBufName));
    config.outputNDWires.push_back(std::make_pair(lhsNDwName, rhsNDwName));
    config.outputBlockers.push_back(
        std::make_pair(lhsBlockerName, rhsBlockerName));

    config.eq.push_back(eqName);

    Attribute attr = lhsFuncOp.getResNames()[i];
    auto strAttr = attr.dyn_cast<StringAttr>();
    // The result name is prefixed with EQ_
    config.results.push_back(
        std::make_pair("EQ_" + strAttr.getValue().str(), lhsResult.getType()));
  }

  // Delete old end operation, we can only have one end operation in a
  // function
  rhsEndOp.erase();
  lhsEndOp.erase();

  // Move operations from lhs to the new miter FuncOp and set the handshake.bb
  for (Operation &op : llvm::make_early_inc_range(lhsFuncOp.getOps())) {
    op.remove();
    newBlock->getOperations().push_back(&op);
    setBB(&op, BB_LHS);
  }

  // Move operations from rhs to the new miter FuncOp and set the handshake.bb
  for (Operation &op : llvm::make_early_inc_range(rhsFuncOp.getOps())) {
    op.remove();
    newBlock->getOperations().push_back(&op);
    setBB(&op, BB_RHS);
  }

  builder.setInsertionPointToEnd(newBlock);
  EndOp newEndOp =
      builder.create<EndOp>(builder.getUnknownLoc(), miterResultValues);
  setHandshakeAttributes(builder, newEndOp, BB_OUT, "end");

  return std::make_pair(miterModule, config);
}

void addCustomCtx(llvm::StringRef rewriteName, ModuleOp modOp) {
  FuncOp funcOp = *modOp.getOps<FuncOp>().begin();
  OpBuilder builder(funcOp.getContext());
  if (rewriteName == "elastic_miter_extension4_lhs_extension4_rhs") {
    llvm::errs() << "Adding custom ctx for extension4_lhs_extension4_rhs\n";
    for (auto passer : funcOp.getOps<PasserOp>()) {
      if (getHandshakeName(passer) == "lhs_passer") {
        builder.setInsertionPoint(passer);
        auto fork =
            builder.create<ForkOp>(builder.getUnknownLoc(), passer.getResult(),
                                   /*numOutputs=*/2);
        setHandshakeAttributes(builder, fork, 1, "ctx_passer_fork");
        passer.getResult().replaceAllUsesExcept(fork->getResult(0), fork);
        auto unconstant = builder.create<UnconstantOp>(builder.getUnknownLoc(),
                                                       fork->getResult(1));
        setHandshakeAttributes(builder, unconstant, 3, "ctx_unconstant");

        NDSourceOp nds = *funcOp.getOps<NDSourceOp>().begin();
        nds.getResult().replaceAllUsesWith(unconstant.getResult());
        nds->erase();
      }
    }
  } else if (rewriteName ==
             "elastic_miter_general_sup_mumux_lhs_general_sup_mumux_rhs") {
    llvm::errs() << "Adding custom ctx for "
                    "general_sup_mumux_lhs_general_sup_mumux_rhs\n";
    builder.setInsertionPointToStart(&funcOp.getBlocks().front());
    Value condArg = funcOp.getArgument(2);
    Value ctrlArg = funcOp.getArgument(3);
    auto condFork = builder.create<ForkOp>(builder.getUnknownLoc(), condArg, 2);
    setHandshakeAttributes(builder, condFork, 0, "ctx_cond_fork");
    condArg.replaceAllUsesExcept(condFork->getResult(0), condFork);
    auto ctrlFork = builder.create<ForkOp>(builder.getUnknownLoc(), ctrlArg, 2);
    setHandshakeAttributes(builder, ctrlFork, 0, "ctx_ctrl_fork");
    ctrlArg.replaceAllUsesExcept(ctrlFork->getResult(0), ctrlFork);
    auto passer = builder.create<PasserOp>(builder.getUnknownLoc(),
                                           condFork->getResult(1),
                                           ctrlFork->getResult(1));
    setHandshakeAttributes(builder, passer, 0, "ctx_passer");
    auto sink =
        builder.create<SinkOp>(builder.getUnknownLoc(), passer.getResult());
    setHandshakeAttributes(builder, sink, 0, "ctx_sink");
  } else if (rewriteName ==
             "elastic_miter_interpInduction_lhs_interpInduction_rhs") {
    llvm::errs()
        << "Adding custom ctx for interpInduction_lhs_interpInduction_rhs\n";
    Value shortArg = funcOp.getArgument(0);
    Value longArg = funcOp.getArgument(1);
    builder.setInsertionPointToStart(&funcOp.getBlocks().front());
    auto shortFork = builder.create<ForkOp>(builder.getUnknownLoc(), shortArg,
                                            /*numOutputs=*/2);
    setHandshakeAttributes(builder, shortFork, 0, "ctx_short_fork");
    shortArg.replaceAllUsesExcept(shortFork->getResult(0), shortFork);
    auto nri = builder.create<SpecV2NDSpeculatorOp>(builder.getUnknownLoc(),
                                                    shortFork->getResult(1));
    setHandshakeAttributes(builder, nri, 0, "ctx_n_repeating_init");
    longArg.replaceAllUsesWith(nri.getResult());
    auto sink = builder.create<SinkOp>(builder.getUnknownLoc(), longArg);
    setHandshakeAttributes(builder, sink, 0, "ctx_sink");
  }
}

FailureOr<std::pair<std::filesystem::path, struct ElasticMiterConfig>>
createMiterFabric(MLIRContext &context, const std::filesystem::path &lhsPath,
                  const std::filesystem::path &rhsPath,
                  const std::filesystem::path &outputDir, size_t nrOfTokens,
                  bool allowNonacceptance, bool timingInsensitive) {
  OwningOpRef<ModuleOp> lhsModuleRef =
      parseSourceFile<ModuleOp>(lhsPath.string(), &context);
  if (!lhsModuleRef) {
    llvm::errs() << "Failed to load LHS module.\n";
    return failure();
  }
  ModuleOp lhsModule = lhsModuleRef.get();

  OwningOpRef<ModuleOp> rhsModuleRef =
      parseSourceFile<ModuleOp>(rhsPath.string(), &context);
  if (!rhsModuleRef) {
    llvm::errs() << "Failed to load RHS module.\n";
    return failure();
  }
  ModuleOp rhsModule = rhsModuleRef.get();

  auto ret = createElasticMiter(context, lhsModule, rhsModule, nrOfTokens,
                                allowNonacceptance, timingInsensitive);
  if (failed(ret)) {
    llvm::errs() << "Failed to create elastic-miter fabric.\n";
    return failure();
  }
  auto [miterModule, config] = ret.value();
  addCustomCtx(config.funcName, miterModule);

  std::string mlirFilename = config.funcName + ".mlir";

  if (failed(createMlirFile(outputDir / mlirFilename, miterModule))) {
    llvm::errs() << "Failed to write MLIR miter file.\n";
    return failure();
  }
  return std::make_pair(outputDir / mlirFilename, config);
}

} // namespace dynamatic::experimental