// Include some other useful headers.
#include "dynamatic/Analysis/NameAnalysis.h" // needed
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h" // needed
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include <algorithm>
#include <fstream>

using namespace llvm;
using namespace dynamatic;

// [START Boilerplate code for the MLIR pass]
#include "experimental/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
namespace experimental {
#define GEN_PASS_DEF_PIPELINEDUPLICATION
#include "experimental/Transforms/Passes.h.inc"
} // namespace experimental
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]
namespace {

struct PipelineDuplicationPass
    : public dynamatic::experimental::impl::PipelineDuplicationBase<
          PipelineDuplicationPass> {

  using PipelineDuplicationBase::PipelineDuplicationBase;

  void runOnOperation() override;

private:
  LogicalResult collectOpsDFS(mlir::Value currentVal,
                              std::vector<mlir::Operation *> endOps,
                              llvm::DenseSet<mlir::Operation *> &visitedOps,
                              llvm::DenseSet<mlir::Value> &outsideDrivers);

  struct PredictionData {
    mlir::Operation *startOp = nullptr;
    mlir::Value predInput;
    std::vector<mlir::Operation *> endOps;
    mlir::ArrayAttr values;
    std::string dataType;
  };

  FailureOr<mlir::ArrayAttr> parseValuesList(mlir::ModuleOp modOp,
                                             llvm::StringRef valuesStr,
                                             std::string &dataType);

  LogicalResult readPredictMarker(mlir::ModuleOp modOp,
                                  std::vector<PredictionData> &pragmaData);
};

} // namespace

/// Performs a Depth-first Search (DFS) from a starting MLIR value to identify
/// all operations for duplication. It traverses the dataflow chain until it
/// either hits a store or a user-defined end operation. Operations that
/// must be duplicated are saved in `visitedOps`, and any results that are used
/// outside of the duplicated region are saved in `outsideDrivers`. The function
/// returns failure if it encounters a dead-end branch before reaching an end
/// operation, indicating an ill-defined duplication graph.
LogicalResult PipelineDuplicationPass::collectOpsDFS(
    mlir::Value currentVal, std::vector<mlir::Operation *> endOps,
    llvm::DenseSet<mlir::Operation *> &visitedOps,
    llvm::DenseSet<mlir::Value> &outsideDrivers) {

  // if a value has no uses, it's a dead end branch without an end operation or
  // store, making the duplicated region ill-defined
  if (currentVal.use_empty()) {
    return failure();
  }

  // for each use of the value
  for (mlir::OpOperand &use : currentVal.getUses()) {
    // get the operation which uses the value
    mlir::Operation *user = use.getOwner();

    // check if already visited
    if (visitedOps.count(user)) {
      continue;
    }

    // stores terminate the duplicated region
    // record the value being stored as an external output driver and skip
    // adding the store operation itself to visitedOps
    if (isa<mlir::memref::StoreOp>(user)) {
      outsideDrivers.insert(currentVal);
      continue;
    }

    visitedOps.insert(user);

    // an end operation is found which is a marker to terminate the duplicated
    // region
    if (llvm::is_contained(endOps, user)) {
      // the results of this operation are added to outsideDrivers
      for (mlir::Value res : user->getResults()) {
        outsideDrivers.insert(res);
      }
      continue;
    }

    // if the operation has no results and is neither and end operation nor a
    // store, this makes the duplicated region ill-defined
    if (user->getNumResults() == 0) {
      return failure();
    }

    // recursive dfs step
    for (mlir::Value result : user->getResults()) {
      if (failed(collectOpsDFS(result, endOps, visitedOps, outsideDrivers))) {
        return failure();
      }
    }
  }
  return success();
}

/// Parse the string containing the predicted values into an ArrayAttr.
FailureOr<mlir::ArrayAttr> PipelineDuplicationPass::parseValuesList(
    mlir::ModuleOp modOp, llvm::StringRef valuesStr, std::string &dataType) {

  // strip potential leading and trailing whitespace
  llvm::StringRef ref = valuesStr.trim();

  // strip brackets
  ref.consume_front("[");
  ref.consume_back("]");

  if (ref.empty()) {
    llvm::errs() << "Error: Value List is empty.\n";
    return failure();
  }

  // split the string by commas into tokens to parse each individual number
  llvm::SmallVector<llvm::StringRef> tokens;
  ref.split(tokens, ',', -1, false);

  llvm::SmallVector<mlir::Attribute> attrValues;
  mlir::OpBuilder builder(modOp.getContext());

  for (auto token : tokens) {
    if (dataType == "double" || dataType == "float") {
      double doubleVal;
      // returns true if casting the string value to double fails
      // otherwise sets doubleVal by reference and returns false
      if (token.getAsDouble(doubleVal)) {
        return failure();
      }
      if (dataType == "float")
        attrValues.push_back(
            builder.getFloatAttr(builder.getF32Type(), doubleVal));
      else
        attrValues.push_back(
            builder.getFloatAttr(builder.getF64Type(), doubleVal));
    } else {
      int intVal;
      // try to parse the string value of token as a base 10 int and store in
      // intVal. returns true if failed, false if succeeded
      if (token.getAsInteger(10, intVal)) {
        return failure();
      }
      if (dataType == "int64_t")
        attrValues.push_back(
            builder.getIntegerAttr(builder.getI64Type(), intVal));
      else
        attrValues.push_back(
            builder.getIntegerAttr(builder.getI32Type(), intVal));
    }
  }
  return (builder.getArrayAttr(attrValues));
}

/// Parses and removes the `dynamatic.prediction_marker` operations from the IR
/// to populate `pragmaData`. It extracts the necessary data from each marker's
/// attributes and identifies the exact start and end operations for each marker
/// before erasing the physical markers from the function.
LogicalResult PipelineDuplicationPass::readPredictMarker(
    mlir::ModuleOp modOp, std::vector<PredictionData> &pragmaData) {

  // maps marker IDs to their corresponding data
  std::map<int, PredictionData> markerMap;
  // tracks all marker operations
  std::vector<mlir::Operation *> markers;

  auto funcOps = modOp.getOps<mlir::func::FuncOp>();
  mlir::func::FuncOp funcOp = *funcOps.begin();

  // find all prediction markers
  for (mlir::Operation &opRef : funcOp.getOps()) {
    mlir::Operation *op = &opRef;
    if (op->getName().getStringRef() != "dynamatic.prediction_marker")
      continue;

    // validate structural expectations of the marker operations
    if (op->getNumOperands() != 1) {
      llvm::errs() << "prediction marker must have exactly one operand";
      return failure();
    }
    if (op->getNumResults() != 1) {
      llvm::errs() << "prediction marker must have exactly one result";
      return failure();
    }

    // forward the marker's input directly to its users, making it possible to
    // safely delete the markers in the end
    // TODO: could this be placed somewhere else?
    op->getResult(0).replaceAllUsesWith(op->getOperand(0));
    markers.push_back(op);
  }

  // process the dynamatic.predict markers
  for (auto op : markers) {

    // extract the dictionary attribute 'dynamatic.predict' which contains the
    // pragma configuration details
    auto predictAttr =
        op->getAttrOfType<mlir::DictionaryAttr>("dynamatic.predict");
    if (!predictAttr) {
      llvm::errs()
          << "Error: This prediction marker doesn't have a dynamatic.predict "
             "attribute containing the pragma configuration details.\n";
      return failure();
    }

    // unpack the specific pragma data fields from the dynamatic.predict
    // dictionary attribute
    auto locationAttr = predictAttr.getAs<mlir::StringAttr>("location");
    auto markerAttr = predictAttr.getAs<mlir::IntegerAttr>("marker");
    auto valuesAttr = predictAttr.getAs<mlir::StringAttr>("values");
    auto typeAttr = predictAttr.getAs<mlir::StringAttr>("type");

    // verify the always necessary attributes (location and marker) were
    // successfully found and parsed
    if (!locationAttr || !markerAttr) {
      llvm::errs() << "Error: Malformed 'dynamatic.predict' location or marker "
                      "attribute.\n";
      return failure();
    }

    // initialize the struct for this marker ID if it doesn't yet exist
    int markerId = markerAttr.getInt();
    if (markerMap.find(markerId) == markerMap.end()) {
      PredictionData newData;

      // verify attributes values and type were successfully found and parsed
      // for a new prediction marker ID. They must exist because the start
      // marker always has to come before any corresponding end markers
      if (!valuesAttr || !typeAttr) {
        llvm::errs() << "Error: Malformed 'dynamatic.predict' values or type "
                        "attribute.\n";
        return failure();
      }

      newData.dataType = typeAttr.getValue();
      auto parsedValues =
          parseValuesList(modOp, valuesAttr.getValue(), newData.dataType);
      if (failed(parsedValues)) {
        llvm::errs() << "Failed to parse value attributes for marker ID "
                     << markerId << '\n';
        return failure();
      }
      newData.values = *parsedValues;

      // insert populated marker into map
      markerMap[markerId] = newData;
    }

    // because of replaceAllUsesWith this is equal to the result
    mlir::Value markerInput = op->getOperand(0);
    llvm::StringRef location = locationAttr.getValue();

    // process the "start" marker to find the starting operation
    if (location == "start") {
      mlir::Operation *nextNode = op->getNextNode();
      bool isUser = llvm::is_contained(nextNode->getOperands(), markerInput);
      if (isUser) {
        markerMap[markerId].startOp = nextNode;
      } else {
        // fall back to the first avilable downstream consumer if the code
        // layout is different than expected (e.g. having the startop right
        // after the marker)
        llvm::errs() << "Warning: Next node does not use the marker. Falling "
                        "back on the next operation that uses it.\n";
        markerMap[markerId].startOp = *markerInput.user_begin();
      }
      markerMap[markerId].predInput = markerInput;

      // process the "end" marker to register the terminating operations of the
      // duplicated region
    } else if (location == "end") {
      // the endop is the definingop of the markerinput
      mlir::Operation *definingOp = markerInput.getDefiningOp();
      markerMap[markerId].endOps.push_back(definingOp);
    }
  }

  // clean up the IR
  for (auto op : markers) {
    op->erase();
  }

  // ensure every collected prediction group has a defined entry point
  for (auto &[op, data] : markerMap) {
    if (!data.startOp) {
      llvm::errs() << "Marker ID " << op
                   << "does not have a valid start operation\n";
      return failure();
    }

    pragmaData.push_back(data);
  }
  return success();
}

void PipelineDuplicationPass::runOnOperation() {
  mlir::ModuleOp modOp = getOperation();
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);
  NameAnalysis &namer = getAnalysis<NameAnalysis>();

  // read input data given in the form of pragmas
  std::vector<PredictionData> pragmaData;
  llvm::errs() << "start reading predict marker data \n";
  if (failed(readPredictMarker(modOp, pragmaData)))
    return signalPassFailure();
  llvm::errs() << "done reading predict marker data \n";

  // print pragmadata
  for (auto data : pragmaData) {
    // llvm::errs() << "Data Type : " << data.dataType << "\n";

    llvm::errs() << "new Values: " << data.values << '\n';

    // Print the start operation
    llvm::errs() << "Start Op  : ";
    if (data.startOp) {
      data.startOp->print(llvm::errs());
    }
    llvm::errs() << "\n";

    llvm::errs() << "predInput : ";
    if (data.predInput) {
      data.predInput.print(llvm::errs());
    }
    llvm::errs() << '\n';

    // Print the end operations
    llvm::errs() << "End Ops   : ";
    if (data.endOps.empty()) {
      llvm::errs() << "None\n";
    } else {
      llvm::errs() << "\n";
      for (mlir::Operation *op : data.endOps) {
        llvm::errs() << "  -> ";
        if (op)
          op->print(llvm::errs());
        else
          llvm::errs() << "nullptr";
        llvm::errs() << "\n";
      }
    }
    llvm::errs() << "------------------------------------\n";
  }

  // iterate over all pragmas with a start
  for (auto data : pragmaData) {

    mlir::Operation *startOp = data.startOp;
    mlir::Value predictInput = data.predInput;
    mlir::Block *targetBlock = startOp->getBlock();

    mlir::func::FuncOp funcOp =
        cast<mlir::func::FuncOp>(targetBlock->getParentOp());
    Location loc = funcOp.getLoc();

    // restructure the blocks
    mlir::Block *exitBlock = targetBlock->splitBlock(startOp);
    mlir::Block *falseBlock = funcOp.addBlock();

    // DFS starting from the result of startop to find all ops that have to be
    // duplicated
    llvm::DenseSet<mlir::Operation *> visitedOps;
    llvm::DenseSet<mlir::Value> outsideDrivers;
    visitedOps.insert(startOp);

    if (data.endOps.empty() ||
        (!data.endOps.empty() && startOp != data.endOps[0])) {
      for (auto op : startOp->getResults()) {
        if (failed(
                collectOpsDFS(op, data.endOps, visitedOps, outsideDrivers))) {
          llvm::errs()
              << "Could not find a valid graph to duplicate. Are all the "
                 "endops placed correctly?\n";
          return signalPassFailure();
        }
      }
      llvm::errs() << "dfs succeeded\n";
    } else
      outsideDrivers.insert(startOp->getResult(0));
    // iterate through the function to sort
    llvm::SmallVector<mlir::Operation *> opsToMove;
    for (mlir::Operation &blockOp : funcOp.getOps()) {
      if (visitedOps.count(&blockOp)) {
        opsToMove.push_back(&blockOp);
      }
    }

    /* for (mlir::Block &block : funcOp.getBlocks()) {
      for (mlir::Operation &blockOp : block.getOperations()) {
        if (visitedOps.count(&blockOp)) {
          opsToMove.push_back(&blockOp);
        }
      }
    } */

    // add values that are needed in next block as arguments to the next block
    llvm::SmallVector<Value> exitBlockArgs;
    llvm::SmallVector<Value> originalOutputs(outsideDrivers.begin(),
                                             outsideDrivers.end());
    for (mlir::Value origOut : originalOutputs) {
      exitBlockArgs.push_back(exitBlock->addArgument(origOut.getType(), loc));
      llvm::errs() << origOut << " origOut\n ";
    }

    // PREDICTED PATHS
    int i = 0;
    for (mlir::Attribute attr : data.values) {
      // create branch condition
      builder.setInsertionPointToEnd(targetBlock);

      Value branchCond, constantComp;
      // does not change for f32 or f64 :)
      if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(attr)) {
        constantComp = builder.create<mlir::arith::ConstantOp>(loc, floatAttr);
        branchCond = builder.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OEQ, predictInput, constantComp);
      } else if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
        constantComp = builder.create<mlir::arith::ConstantOp>(loc, intAttr);
        branchCond = builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::eq, predictInput, constantComp);
      } else {
        return signalPassFailure();
      }

      mlir::Block *trueBlock = funcOp.addBlock(); // true path
      mlir::Block *nextElseBlock;                 // false path
      if (i + 1 < (int)data.values.size())
        nextElseBlock = funcOp.addBlock();
      else
        nextElseBlock = falseBlock;

      builder.create<mlir::cf::CondBranchOp>(loc, branchCond, trueBlock,
                                             nextElseBlock);

      // move the new blocks to right after targetBlock
      auto &blockList = funcOp.getBody().getBlocks();
      blockList.splice(std::next(targetBlock->getIterator()), blockList,
                       trueBlock->getIterator());
      blockList.splice(std::next(trueBlock->getIterator()), blockList,
                       nextElseBlock->getIterator());

      // clone the necessary operations to here
      builder.setInsertionPointToStart(trueBlock);
      mlir::IRMapping mapper;
      mapper.map(predictInput, constantComp);

      // do the actual cloning
      for (mlir::Operation *origOp : opsToMove) {
        mlir::Operation *cloned = builder.clone(*origOp, mapper);
        std::string newName =
            namer.getName(origOp).str() + "_dup" + std::to_string(i);
        cloned->setAttr("handshake.name", builder.getStringAttr(newName));

        for (unsigned j = 0; j < cloned->getNumResults(); j++) {
          // make sure cloned operations go to cloned operations
          mapper.map(origOp->getResult(j), cloned->getResult(j));
        }
      }

      // Print Value mappings (Original Value -> Cloned Value)
      llvm::errs() << "--- Mapper Contents ---\n";
      for (auto &pair : mapper.getValueMap()) {
        mlir::Value original = pair.first;
        mlir::Value cloned = pair.second;

        llvm::errs() << "Value Mapping:\n";
        llvm::errs() << "  From: " << original << "\n";
        llvm::errs() << "  To:   " << cloned << "\n";
      }
      llvm::errs() << "-----------------------\n";

      llvm::SmallVector<Value> trueBranchOperands;
      for (Value origOut : originalOutputs) {
        trueBranchOperands.push_back(mapper.lookup(origOut));
      }
      builder.setInsertionPointToEnd(trueBlock);
      builder.create<mlir::cf::BranchOp>(loc, exitBlock, trueBranchOperands);

      // update
      targetBlock = nextElseBlock;
      i++;
    }

    // FALSE PATH
    // move all of the stuff from above here
    builder.setInsertionPointToStart(targetBlock);
    for (Operation *origOp : opsToMove) {
      origOp->moveBefore(targetBlock, targetBlock->end());
    }
    builder.setInsertionPointToEnd(targetBlock);
    builder.create<mlir::cf::BranchOp>(loc, exitBlock, originalOutputs);

    // make sure exitBlock reads its stuff from block arguments instead of old
    // operations that do not exist anymore in that sense
    for (size_t i = 0; i < originalOutputs.size(); ++i) {
      llvm::errs() << originalOutputs[i] << '\n' << exitBlockArgs[i] << '\n';
      originalOutputs[i].replaceUsesWithIf(
          exitBlockArgs[i], [&](OpOperand &operand) {
            mlir::Block *userBlock = operand.getOwner()->getBlock();
            if (userBlock == exitBlock) {
              return true;
            }

            // Otherwise, only replace it if it's completely outside the
            // true/false cascade structures we generated.
            // TODO: this is wrong
            bool isCondBranch =
                userBlock->getTerminator() &&
                isa<mlir::cf::CondBranchOp>(userBlock->getTerminator());

            llvm::errs() << isCondBranch << " condbranch\n";

            return userBlock != targetBlock &&
                   userBlock->getParentOp() == funcOp && isCondBranch;
          });
    }
  }

  // Automatically run push-constants pass after duplication
  mlir::PassManager pm(ctx);
  pm.addPass(dynamatic::createPushConstants());
  if (failed(pm.run(modOp)))
    return signalPassFailure();
}
