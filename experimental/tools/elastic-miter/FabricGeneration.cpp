//===- FabricGeneration.cpp ------------------------------------ *- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AsmState.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <filesystem>
#include <iterator>
#include <optional>
#include <string>
#include <utility>

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
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

static inline bool eligibleForMaterialization(Value val) {
  return isa<handshake::ControlType, handshake::ChannelType>(val.getType());
}

LogicalResult verifyIRMaterializedExceptForArgs(handshake::FuncOp funcOp) {
  auto checkUses = [&](Operation *op, Value val, StringRef desc,
                       unsigned idx) -> LogicalResult {
    if (!eligibleForMaterialization(val))
      return success();

    auto numUses = std::distance(val.getUses().begin(), val.getUses().end());
    if (numUses == 0)
      return op->emitError() << desc << " " << idx << " has no uses.";
    if (numUses > 1)
      return op->emitError() << desc << " " << idx << " has multiple uses.";
    return success();
  };

  // Check results of operations
  for (Operation &op : llvm::make_early_inc_range(funcOp.getOps())) {
    for (OpResult res : op.getResults()) {
      if (failed(checkUses(&op, res, "result", res.getResultNumber())))
        return failure();
    }
  }
  return success();
}

// Get the first and only non-external FuncOp from the module.
// Additionally check some properites:
// 1. The FuncOp is materialized (each Value is only used once).
// 2. There are no memory interfaces
// 3. Arguments  and results are all handshake.channel or handshake.control type
FailureOr<FuncOp> getModuleFuncOpAndCheck(ModuleOp module,
                                          bool allowArgsNonmaterialization) {
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
    if (allowArgsNonmaterialization) {
      if (failed(verifyIRMaterializedExceptForArgs(funcOp))) {
        llvm::errs() << dynamatic::ERR_NON_MATERIALIZED_FUNC;
        return failure();
      }
    } else {
      if (failed(dynamatic::verifyIRMaterialized(funcOp))) {
        llvm::errs() << dynamatic::ERR_NON_MATERIALIZED_FUNC;
        return failure();
      }
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

llvm::StringMap<Type> getAllInputValuesFromMiterContext(FuncOp contextFunc) {
  auto endOps = contextFunc.getOps<EndOp>();
  assert(std::distance(endOps.begin(), endOps.end()) == 1 &&
         "There should be exactly one EndOp in the context function");
  EndOp endOp = *endOps.begin();

  llvm::StringMap<Type> inputValues;
  for (auto [i, result] : llvm::enumerate(endOp.getOperands())) {
    // Get the name of the result
    auto resName = contextFunc.getResName(i);
    // Get the type of the result
    auto resType = result.getType();
    // Add to the map
    inputValues.insert({resName.str(), resType});
  }
  return inputValues;
}

std::optional<BlockArgument> findArg(FuncOp funcOp, StringRef argName) {
  for (auto [i, attr] : llvm::enumerate(funcOp.getArgNames())) {
    auto strAttr = attr.dyn_cast<StringAttr>();
    if (strAttr && strAttr.getValue() == argName) {
      return funcOp.getArgument(i);
    }
  }

  return std::nullopt;
}

ForkOp flattenFork(ForkOp topFork) {
  SmallVector<Value> values;
  SmallVector<ForkOp> forksToBeErased;
  for (OpResult result : llvm::make_early_inc_range(topFork.getResults())) {
    materializeValue(result);
    Operation *user = getUniqueUser(result);

    if (auto forkOp = dyn_cast<ForkOp>(user)) {
      ForkOp newForkOp = flattenFork(forkOp);
      for (OpResult forkResult : newForkOp.getResults()) {
        assertMaterialization(forkResult);
        values.push_back(forkResult);
      }
      forksToBeErased.push_back(newForkOp);
    } else {
      values.push_back(result);
    }
  }

  if (values.size() == topFork.getResults().size())
    return topFork; // No need to flatten, already flat

  OpBuilder builder(topFork->getContext());
  builder.setInsertionPoint(topFork);
  ForkOp newForkOp = builder.create<ForkOp>(
      topFork.getLoc(), topFork.getOperand(), values.size());
  inheritBB(topFork, newForkOp);

  for (unsigned i = 0; i < values.size(); ++i) {
    values[i].replaceAllUsesWith(newForkOp.getResult()[i]);
  }

  for (ForkOp forkOp : forksToBeErased) {
    // Erase the old fork
    forkOp->erase();
  }
  topFork->erase();

  return newForkOp;
}

void materializeValue(Value val) {
  Operation *definingOp = val.getDefiningOp();
  OpBuilder builder(definingOp->getContext());
  builder.setInsertionPoint(definingOp);

  if (val.use_empty()) {
    SinkOp sinkOp = builder.create<SinkOp>(val.getLoc(), val);
    inheritBB(definingOp, sinkOp);
    return;
  }
  if (val.hasOneUse())
    return;

  unsigned numUses = std::distance(val.getUses().begin(), val.getUses().end());

  ForkOp forkOp = builder.create<ForkOp>(val.getLoc(), val, numUses);
  inheritBB(definingOp, forkOp);

  int i = 0;
  // To allow operand mutation, we use an early-increment range.
  // TODO: The original author may have been unaware of this approach;
  // the materialization pass appears unnecessarily complex. Consider
  // refactoring it to use an early-increment range as well.
  for (OpOperand &opOperand : llvm::make_early_inc_range(val.getUses())) {
    if (opOperand.getOwner() == forkOp)
      continue;
    opOperand.set(forkOp.getResult()[i]);
    i++;
  }

  flattenFork(forkOp);
}

void materializeArg(BlockArgument val, unsigned bb) {
  OpBuilder builder(val.getContext());
  builder.setInsertionPointToStart(val.getParentBlock());

  if (val.use_empty()) {
    SinkOp sinkOp = builder.create<SinkOp>(val.getLoc(), val);
    setBB(sinkOp, bb);
    return;
  }
  if (val.hasOneUse())
    return;

  unsigned numUses = std::distance(val.getUses().begin(), val.getUses().end());

  ForkOp forkOp = builder.create<ForkOp>(val.getLoc(), val, numUses);
  setBB(forkOp, bb);

  int i = 0;
  // To allow operand mutation, we use an early-increment range.
  // TODO: The original author may have been unaware of this approach;
  // the materialization pass appears unnecessarily complex. Consider
  // refactoring it to use an early-increment range as well.
  for (OpOperand &opOperand : llvm::make_early_inc_range(val.getUses())) {
    if (opOperand.getOwner() == forkOp)
      continue;
    opOperand.set(forkOp.getResult()[i]);
    i++;
  }

  flattenFork(forkOp);
}

Operation *getUniqueUser(Value val) {
  assertMaterialization(val);
  return *val.getUsers().begin();
}

void assertMaterialization(Value val) {
  if (val.hasOneUse())
    return;
  val.getDefiningOp()->emitError("Expected the value to be materialized, but "
                                 "it has zero or multiple users");
  llvm_unreachable("MaterializationUtil failed");
}

FuncOp fuseValueManager(FuncOp contextFuncOp, FuncOp funcOp) {
  OpBuilder builder(funcOp.getContext());

  auto [newFuncOp, newBlock] = buildNewFuncWithBlock(
      funcOp.getName(), contextFuncOp.getResultTypes(), funcOp.getResultTypes(),
      contextFuncOp.getResNames(), funcOp.getResNames());

  funcOp->getParentOfType<ModuleOp>().push_back(newFuncOp);

  builder.setInsertionPointToStart(newBlock);

  for (auto [i, newArg] : llvm::enumerate(newFuncOp.getArguments())) {
    // Get the name and type of the input value
    auto name = contextFuncOp.getArgName(i);

    if (auto arg = findArg(funcOp, name)) {
      assert(arg->getType() == newArg.getType() &&
             "The type of the argument does not match the expected type");

      arg->replaceAllUsesWith(newArg);
    }

    materializeArg(newArg, 1);
  }

  for (Operation &op : llvm::make_early_inc_range(funcOp.getOps())) {
    op.remove();
    newBlock->getOperations().push_back(&op);
    unsigned bb = getLogicBB(&op).value_or(0);
    setBB(&op, bb + 1);
  }

  funcOp->erase();
  return newFuncOp;
}

std::pair<FuncOp, unsigned> fuseContext(FuncOp contextFuncOp, FuncOp funcOp) {
  OpBuilder builder(funcOp.getContext());

  auto [newFuncOp, newBlock] =
      buildNewFuncWithBlock(funcOp.getName(), contextFuncOp.getArgumentTypes(),
                            funcOp.getResultTypes(),
                            contextFuncOp.getArgNames(), funcOp.getResNames());

  funcOp->getParentOfType<ModuleOp>().push_back(newFuncOp);

  builder.setInsertionPointToStart(newBlock);

  for (auto [i, newArg] : llvm::enumerate(newFuncOp.getArguments())) {
    auto oldArg = contextFuncOp.getArgument(i);
    oldArg.replaceAllUsesWith(newArg);
  }

  auto endOps = contextFuncOp.getOps<EndOp>();
  assert(std::distance(endOps.begin(), endOps.end()) == 1 &&
         "There should be exactly one EndOp in the context function");
  EndOp endOp = *endOps.begin();

  for (auto [i, oldArg] : llvm::enumerate(funcOp.getArguments())) {
    auto newResult = endOp->getOperand(i);
    oldArg.replaceAllUsesWith(newResult);
  }
  endOp->erase();

  // move all operations from the context function to the new function
  unsigned contextMaxBB = 0;
  for (Operation &op : llvm::make_early_inc_range(contextFuncOp.getOps())) {
    op.remove();
    newBlock->getOperations().push_back(&op);
    unsigned bb = getLogicBB(&op).value_or(0);
    setBB(&op, bb + 1);
    contextMaxBB = std::max(contextMaxBB, bb + 1);
  }

  unsigned maxBB = contextMaxBB;
  for (Operation &op : llvm::make_early_inc_range(funcOp.getOps())) {
    op.remove();
    newBlock->getOperations().push_back(&op);
    unsigned bb = getLogicBB(&op).value_or(0);
    setBB(&op, bb + contextMaxBB);
    maxBB = std::max(maxBB, bb + contextMaxBB);
  }

  contextFuncOp->erase();
  funcOp->erase();

  return std::make_pair(newFuncOp, maxBB);
}

// Create the reachability circuit by putting ND wires at all the in- and
// outputs
FailureOr<std::pair<ModuleOp, struct ElasticMiterConfig>>
createReachabilityCircuit(MLIRContext &context,
                          const std::filesystem::path &filename,
                          const std::filesystem::path &contextPath) {

  OwningOpRef<ModuleOp> mod =
      parseSourceFile<ModuleOp>(filename.string(), &context);
  if (!mod)
    return failure();

  OwningOpRef<ModuleOp> contextMod =
      parseSourceFile<ModuleOp>(contextPath.string(), &context);
  if (!contextMod)
    return failure();
  auto contextFuncOrFailure = getModuleFuncOpAndCheck(contextMod.get(), false);
  if (failed(contextFuncOrFailure)) {
    llvm::errs() << "Failed to get context FuncOp.\n";
    return failure();
  }

  // Get the FuncOp from the module, also check some needed properties
  auto funcOrFailure = getModuleFuncOpAndCheck(mod.get(), true);
  if (failed(funcOrFailure))
    return failure();
  FuncOp funcOp = funcOrFailure.value();

  funcOp = fuseValueManager(contextFuncOrFailure.value(), funcOp);

  mod->dump();

  auto [newFuncOp, maxBB] = fuseContext(contextFuncOrFailure.value(), funcOp);
  funcOp = newFuncOp;

  mod->dump();

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

    std::string ndwName = "ndw_in_" + funcOp.getArgName(i).str();

    NDWireOp ndWireOp = builder.create<NDWireOp>(funcOp.getLoc(), arg);
    setHandshakeAttributes(builder, ndWireOp, 0, ndwName);

    // Use the newly created NDwire's output instead of the original argument in
    // the funcOp's operations
    for (Operation *op : llvm::make_early_inc_range(arg.getUsers())) {
      if (op != ndWireOp)
        op->replaceUsesOfWith(arg, ndWireOp.getResult());
    }

    Attribute attr = funcOp.getArgNames()[i];
    auto strAttr = attr.dyn_cast<StringAttr>();
    config.arguments.push_back(
        std::make_pair(strAttr.getValue().str(), arg.getType()));
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
  for (unsigned i = 0; i < endOp.getOperands().size(); ++i) {
    Value result = endOp.getOperand(i);

    std::string ndwName = "ndw_out_" + funcOp.getResName(i).str();

    NDWireOp endNDWireOp = builder.create<NDWireOp>(endOp->getLoc(), result);
    setHandshakeAttributes(builder, endNDWireOp, maxBB + 1, ndwName);

    // Use the newly created NDwire's output instead of the original argument in
    // the FuncOp's operations
    for (Operation *op : llvm::make_early_inc_range(result.getUsers())) {
      if (op != endNDWireOp)
        op->replaceUsesOfWith(result, endNDWireOp.getResult());
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
                   size_t bufferSlots, bool ndSpec, bool allowNonacceptance) {

  // Get the LHS FuncOp from the LHS module, also check some required properties
  auto funcOrFailure = getModuleFuncOpAndCheck(lhsModule, true);
  if (failed(funcOrFailure)) {
    llvm::errs() << "Failed to get LHS FuncOp.\n";
    return failure();
  }
  FuncOp lhsFuncOp = funcOrFailure.value();

  // Get the RHS FuncOp from the RHS module
  funcOrFailure = getModuleFuncOpAndCheck(rhsModule, true);
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

  Operation *nextLocation;

  if (ndSpec) {
    assert(lhsFuncOp.getNumArguments() == 2 &&
           "The ND speculator context only supports two arguments.");
  }

  // Create the input side auxillary logic:
  // Every primary input of the LHS/RHS circuits is connected to a fork, which
  // splits the data for both sides under test. Both output sides of the fork
  // are connected to a buffer. The output of the buffer is connected to a
  // non-deterministic wire (ND wire), which models arbitrary backpressure
  // from surrounding circuits. The output of the ND wire is then connected to
  // the original input of the circuits. This completely decouples the
  // operation of the two circuits.
  for (unsigned i = 0; i < lhsFuncOp.getNumArguments(); ++i) {
    Value lhsArg = lhsFuncOp.getArgument(i);
    Value rhsArg = rhsFuncOp.getArgument(i);
    Value miterArg = newFuncOp.getArgument(i);
    if (getHandshakeTypeBitWidth(miterArg.getType()) > 1) {
      return lhsFuncOp.emitError(
          "ElasticMiter currently supports only 1-bit or "
          "control inputs.");
    }

    if (ndSpec && i == 1) {
      LazyForkOp forkOp = builder.create<LazyForkOp>(
          newFuncOp.getLoc(), newFuncOp.getArgument(0), 2);
      setHandshakeAttributes(builder, forkOp, BB_IN, "in_nd_fork");
      newFuncOp.getArgument(0).replaceAllUsesExcept(forkOp.getResult()[0],
                                                    forkOp);

      BufferOp ndSpecPreBufferOp = builder.create<BufferOp>(
          forkOp.getLoc(), forkOp.getResult()[1], bufferSlots,
          dynamatic::handshake::BufferType::FIFO_BREAK_DV);
      setHandshakeAttributes(builder, ndSpecPreBufferOp, BB_IN,
                             "in_nd_spec_pre_buffer");
      SpecV2NDSpeculatorOp ndSpecOp = builder.create<SpecV2NDSpeculatorOp>(
          newFuncOp.getLoc(), ndSpecPreBufferOp.getResult());
      setHandshakeAttributes(builder, ndSpecOp, BB_IN, "in_nd_speculator");
      SpecV2RepeatingInitOp riOp = builder.create<SpecV2RepeatingInitOp>(
          newFuncOp.getLoc(), ndSpecOp.getResult());
      setHandshakeAttributes(builder, riOp, BB_IN, "in_ri");
      miterArg = riOp.getResult();

      SinkOp sinkOp =
          builder.create<SinkOp>(newFuncOp.getLoc(), newFuncOp.getArgument(1));
      setHandshakeAttributes(builder, sinkOp, BB_IN, "in_sink");
    }

    std::string forkName = "in_fork_" + lhsFuncOp.getArgName(i).str();
    std::string lhsBufName = "lhs_in_buf_" + lhsFuncOp.getArgName(i).str();
    std::string rhsBufName = "rhs_in_buf_" + lhsFuncOp.getArgName(i).str();
    std::string lhsNdwName = "lhs_in_ndw_" + lhsFuncOp.getArgName(i).str();
    std::string rhsNdwName = "rhs_in_ndw_" + lhsFuncOp.getArgName(i).str();

    LazyForkOp forkOp =
        builder.create<LazyForkOp>(newFuncOp.getLoc(), miterArg, 2);
    setHandshakeAttributes(builder, forkOp, BB_IN, forkName);

    BufferOp lhsBufferOp = builder.create<BufferOp>(
        forkOp.getLoc(), forkOp.getResult()[0], bufferSlots,
        dynamatic::handshake::BufferType::FIFO_BREAK_DV);
    BufferOp rhsBufferOp = builder.create<BufferOp>(
        forkOp.getLoc(), forkOp.getResult()[1], bufferSlots,
        dynamatic::handshake::BufferType::FIFO_BREAK_DV);
    setHandshakeAttributes(builder, lhsBufferOp, BB_IN, lhsBufName);
    setHandshakeAttributes(builder, rhsBufferOp, BB_IN, rhsBufName);

    NDWireOp lhsNDWireOp =
        builder.create<NDWireOp>(forkOp.getLoc(), lhsBufferOp.getResult());
    NDWireOp rhsNDWireOp =
        builder.create<NDWireOp>(forkOp.getLoc(), rhsBufferOp.getResult());
    setHandshakeAttributes(builder, lhsNDWireOp, BB_IN, lhsNdwName);
    setHandshakeAttributes(builder, rhsNDWireOp, BB_IN, rhsNdwName);

    nextLocation = rhsNDWireOp;

    config.inputBuffers.push_back(std::make_pair(lhsBufName, rhsBufName));
    config.inputNDWires.push_back(std::make_pair(lhsNdwName, rhsNdwName));

    // Use the newly created fork's output instead of the original argument in
    // the lhsFuncOp's operations
    for (Operation *op : llvm::make_early_inc_range(lhsArg.getUsers()))
      op->replaceUsesOfWith(lhsArg, lhsNDWireOp.getResult());

    // Use the newly created fork's output instead of the original argument in
    // the rhsFuncOp's operations
    for (Operation *op : llvm::make_early_inc_range(rhsArg.getUsers()))
      op->replaceUsesOfWith(rhsArg, rhsNDWireOp.getResult());

    Attribute attr = lhsFuncOp.getArgNames()[i];
    auto strAttr = attr.dyn_cast<StringAttr>();
    config.arguments.push_back(
        std::make_pair(strAttr.getValue().str(), lhsArg.getType()));
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
  llvm::SmallVector<Value> miterResultValues;
  for (unsigned i = 0; i < lhsEndOp.getOperands().size(); ++i) {
    Value lhsResult = lhsEndOp.getOperand(i);
    Value rhsResult = rhsEndOp.getOperand(i);

    std::string outName = lhsFuncOp.getResName(i).str();
    std::string lhsBufName = "lhs_out_buf_" + lhsFuncOp.getResName(i).str();
    std::string rhsBufName = "rhs_out_buf_" + lhsFuncOp.getResName(i).str();
    std::string lhsNDwName = "lhs_out_ndw_" + lhsFuncOp.getResName(i).str();
    std::string rhsNDwName = "rhs_out_ndw_" + lhsFuncOp.getResName(i).str();
    std::string eqName = "out_eq_" + lhsFuncOp.getResName(i).str();

    if (allowNonacceptance) {
      NDSourceOp ndSourceOp =
          builder.create<NDSourceOp>(nextLocation->getLoc());
      setHandshakeAttributes(builder, ndSourceOp, BB_OUT, "out_nds_" + outName);
      LazyForkOp lazyForkOp = builder.create<LazyForkOp>(
          nextLocation->getLoc(), ndSourceOp.getResult(), 2);
      setHandshakeAttributes(builder, lazyForkOp, BB_OUT, "out_lf_" + outName);

      BufferOp lhsNDSBufferOp = builder.create<BufferOp>(
          nextLocation->getLoc(), lazyForkOp.getResults()[0], bufferSlots,
          dynamatic::handshake::BufferType::FIFO_BREAK_DV);
      setHandshakeAttributes(builder, lhsNDSBufferOp, BB_OUT,
                             "out_buf_lhs_nds_" + outName);
      BufferOp rhsNDSBufferOp = builder.create<BufferOp>(
          nextLocation->getLoc(), lazyForkOp.getResults()[1], bufferSlots,
          dynamatic::handshake::BufferType::FIFO_BREAK_DV);
      setHandshakeAttributes(builder, rhsNDSBufferOp, BB_OUT,
                             "out_buf_rhs_nds_" + outName);

      BlockerOp lhsBlockerOp = builder.create<BlockerOp>(
          nextLocation->getLoc(), lhsResult, lhsNDSBufferOp.getResult());
      setHandshakeAttributes(builder, lhsBlockerOp, BB_OUT,
                             "out_lhs_bl_" + outName);
      BlockerOp rhsBlockerOp = builder.create<BlockerOp>(
          nextLocation->getLoc(), rhsResult, rhsNDSBufferOp.getResult());
      setHandshakeAttributes(builder, rhsBlockerOp, BB_OUT,
                             "out_rhs_bl_" + outName);

      lhsResult = lhsBlockerOp.getResult();
      rhsResult = rhsBlockerOp.getResult();
    }

    NDWireOp lhsEndNDWireOp;
    NDWireOp rhsEndNDWireOp;

    lhsEndNDWireOp =
        builder.create<NDWireOp>(nextLocation->getLoc(), lhsResult);
    rhsEndNDWireOp =
        builder.create<NDWireOp>(nextLocation->getLoc(), rhsResult);

    setHandshakeAttributes(builder, lhsEndNDWireOp, BB_OUT, lhsNDwName);
    setHandshakeAttributes(builder, rhsEndNDWireOp, BB_OUT, rhsNDwName);

    BufferOp lhsEndBufferOp = builder.create<BufferOp>(
        nextLocation->getLoc(), lhsEndNDWireOp.getResult(), bufferSlots,
        dynamatic::handshake::BufferType::FIFO_BREAK_DV);
    BufferOp rhsEndBufferOp = builder.create<BufferOp>(
        nextLocation->getLoc(), rhsEndNDWireOp.getResult(), bufferSlots,
        dynamatic::handshake::BufferType::FIFO_BREAK_DV);

    setHandshakeAttributes(builder, lhsEndBufferOp, BB_OUT, lhsBufName);
    setHandshakeAttributes(builder, rhsEndBufferOp, BB_OUT, rhsBufName);

    if (lhsResult.getType().isa<handshake::ControlType>()) {
      SmallVector<Value> joinInputs = {lhsEndBufferOp.getResult(),
                                       rhsEndBufferOp.getResult()};
      JoinOp joinOp =
          builder.create<JoinOp>(builder.getUnknownLoc(), joinInputs);
      setHandshakeAttributes(builder, joinOp, BB_OUT, eqName);
      miterResultValues.push_back(joinOp.getResult());
    } else {
      CmpIOp compOp = builder.create<CmpIOp>(
          builder.getUnknownLoc(), CmpIPredicate::eq,
          lhsEndBufferOp.getResult(), rhsEndBufferOp.getResult());
      setHandshakeAttributes(builder, compOp, BB_OUT, eqName);
      miterResultValues.push_back(compOp.getResult());
    }

    config.outputBuffers.push_back(std::make_pair(lhsBufName, rhsBufName));
    config.outputNDWires.push_back(std::make_pair(lhsNDwName, rhsNDwName));

    config.eq.push_back(eqName);

    Attribute attr = lhsFuncOp.getResNames()[i];
    auto strAttr = attr.dyn_cast<StringAttr>();
    // The result name is prefixed with EQ_
    config.results.push_back(
        std::make_pair("EQ_" + strAttr.getValue().str(), lhsResult.getType()));
  }

  EndOp newEndOp =
      builder.create<EndOp>(builder.getUnknownLoc(), miterResultValues);
  setHandshakeAttributes(builder, newEndOp, BB_OUT, "end");

  // Delete old end operation, we can only have one end operation in a
  // function
  rhsEndOp.erase();
  lhsEndOp.erase();

  // Move operations from lhs to the new miter FuncOp and set the handshake.bb
  Operation *previousOp = nextLocation;
  for (Operation &op : llvm::make_early_inc_range(lhsFuncOp.getOps())) {
    op.moveAfter(previousOp);
    dynamatic::setBB(&op, BB_LHS);
    previousOp = &op;
  }

  // Move operations from rhs to the new miter FuncOp and set the handshake.bb
  for (Operation &op : llvm::make_early_inc_range(rhsFuncOp.getOps())) {
    op.moveAfter(previousOp);
    dynamatic::setBB(&op, BB_RHS);
    previousOp = &op;
  }

  return std::make_pair(miterModule, config);
}

FailureOr<std::pair<std::filesystem::path, struct ElasticMiterConfig>>
createMiterFabric(MLIRContext &context, const std::filesystem::path &lhsPath,
                  const std::filesystem::path &rhsPath,
                  const std::filesystem::path &outputDir, size_t nrOfTokens,
                  bool ndSpec, bool allowNonacceptance) {

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
                                ndSpec, allowNonacceptance);
  if (failed(ret)) {
    llvm::errs() << "Failed to create elastic-miter fabric.\n";
    return failure();
  }
  auto [miterModule, config] = ret.value();

  std::string mlirFilename = config.funcName + ".mlir";

  if (failed(createMlirFile(outputDir / mlirFilename, miterModule))) {
    llvm::errs() << "Failed to write MLIR miter file.\n";
    return failure();
  }
  return std::make_pair(outputDir / mlirFilename, config);
}

FailureOr<llvm::StringMap<Type>>
analyzeInputValue(MLIRContext &context, const std::filesystem::path &path) {
  OwningOpRef<ModuleOp> mod =
      parseSourceFile<ModuleOp>(path.string(), &context);
  if (!mod)
    return failure();

  auto funcOps = mod.get().getOps<FuncOp>();
  assert(std::distance(funcOps.begin(), funcOps.end()) == 1 &&
         "Expected a single function");

  FuncOp funcOp = *funcOps.begin();
  llvm::StringMap<Type> inputValues;
  OpPrintingFlags defaultFlag;
  auto argNames = funcOp.getArgNames();
  for (auto [i, blockArg] : llvm::enumerate(funcOp.getArguments())) {
    inputValues.insert(
        {argNames[i].cast<StringAttr>().getValue(),
         blockArg.getType()}); // Remove the leading '%' from the name
  }

  return inputValues;
}

LogicalResult
generateDefaultMiterContext(MLIRContext &context,
                            const llvm::StringMap<mlir::Type> &allInputValues,
                            const std::filesystem::path &outputFile) {
  OpBuilder builder(&context);
  ModuleOp module = ModuleOp::create(builder.getUnknownLoc());

  // Create a new function with the input values as arguments
  SmallVector<Attribute> argNames;
  SmallVector<Type> argTypes;

  for (const auto &pair : allInputValues) {
    argNames.push_back(builder.getStringAttr(pair.first()));
    argTypes.push_back(pair.second);
  }

  auto argAttr = builder.getArrayAttr(argNames);
  auto resAttr = builder.getArrayAttr(argNames);
  auto [funcOp, entryBlock] =
      buildNewFuncWithBlock("context", argTypes, argTypes, argAttr, resAttr);
  module.push_back(funcOp);

  builder.setInsertionPointToStart(entryBlock);

  EndOp endOp =
      builder.create<EndOp>(builder.getUnknownLoc(), funcOp.getArguments());
  setHandshakeAttributes(builder, endOp, 1, "end");

  // Write the module to the output file
  if (failed(createMlirFile(outputFile, module))) {
    llvm::errs() << "Failed to write default context MLIR file.\n";
    return failure();
  }

  return success();
}

} // namespace dynamatic::experimental