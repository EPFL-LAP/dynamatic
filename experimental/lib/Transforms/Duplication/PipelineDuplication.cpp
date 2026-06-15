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

// NOTE: The code wrapped in LLVM_DEBUG(...) is executed when
// - Dynamatic is built in debug mode
// - dynamatic-opt is called with `--debug` or `--debug-only=<DEBUG_TYPE>`.
#define DEBUG_TYPE "pipeline-duplication"

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
                              const std::vector<mlir::Operation *> &endOps,
                              mlir::Block *currentBlock,
                              llvm::DenseSet<mlir::Operation *> &visitedOps);

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

  void
  collectDependenciesUpstream(mlir::Operation *op, mlir::Block *targetBlock,
                              llvm::DenseSet<mlir::Operation *> &visitedOps);

  FailureOr<
      std::pair<llvm::DenseSet<mlir::Block *>, llvm::DenseSet<mlir::Block *>>>
  validateDuplicationStructure(
      const llvm::DenseSet<mlir::Operation *> &visitedOps,
      const std::vector<mlir::Operation *> &endOps);
};

} // namespace

/// Validate whether the structure can be duplicated and is well-defined
FailureOr<
    std::pair<llvm::DenseSet<mlir::Block *>, llvm::DenseSet<mlir::Block *>>>
PipelineDuplicationPass::validateDuplicationStructure(
    const llvm::DenseSet<mlir::Operation *> &visitedOps,
    const std::vector<mlir::Operation *> &endOps) {

  // which blocks are involved
  llvm::DenseSet<mlir::Block *> duplicatedBlocks;
  // in which blocks the duplication stops
  llvm::DenseSet<mlir::Block *> terminationBlocks;

  // find all blocks that are involved
  for (mlir::Operation *op : visitedOps) {
    mlir::Block *bb = op->getBlock();
    llvm::errs() << "a duplicated block: " << bb << '\n';
    duplicatedBlocks.insert(bb);

    // end ops and store ops mark that the duplication stops
    if (llvm::is_contained(endOps, op) || isa<mlir::memref::StoreOp>(op)) {
      llvm::errs() << "a terminating block: " << bb << '\n';
      terminationBlocks.insert(bb);
    }
  }

  llvm::errs() << "size of terminationBlocks: " << terminationBlocks.size();

  // verify that there are no espacing paths from the termination blocks
  for (mlir::Block *termBlock : terminationBlocks) {
    for (mlir::Block *successor : termBlock->getSuccessors()) {
      // if a block has a duplication terminator (endop), it cannot feed into
      // another duplicated block
      if (duplicatedBlocks.count(successor)) {
        llvm::errs() << "Escaping paths from termination blocks found.\n";
        return failure();
      }
    }
  }

  // verify that all termination blocks converge to one single point
  llvm::DenseSet<mlir::Block *> externalSuccessors;
  for (mlir::Block *dupBlock : duplicatedBlocks) {
    for (mlir::Block *successor : dupBlock->getSuccessors()) {
      // find the blocks that lead to a nonduplicated block
      if (!duplicatedBlocks.count(successor)) {
        llvm::errs() << "blocks that lead to a nonduplicated block:" << dupBlock
                     << "and its successor:" << successor << '\n';
        externalSuccessors.insert(successor);
      }
    }
  }

  // if the cloned blocks exit to multiple distinct external blocks, the graph
  // doesn't reconverge correctly
  if (externalSuccessors.size() > 1 && duplicatedBlocks.size() > 1) {
    llvm::errs() << "More than one successor of the termination blocks.\n";
    return failure();
  }

  return {{duplicatedBlocks, terminationBlocks}};
}

/// Helper function to backward-traverse and collect all dependencies coming
/// from the same block
void PipelineDuplicationPass::collectDependenciesUpstream(
    mlir::Operation *op, mlir::Block *targetBlock,
    llvm::DenseSet<mlir::Operation *> &visitedOps) {

  if (!op || op->getBlock() != targetBlock) {
    return;
  }

  for (mlir::Value operand : op->getOperands()) {
    if (mlir::Operation *defOp = operand.getDefiningOp()) {
      // Only care about operations within the same basic block
      if (defOp->getBlock() == targetBlock) {
        // attempt to insert the op into our set of visited operations
        // returns true only if the element was not already present in the set
        if (visitedOps.insert(defOp).second) {
          collectDependenciesUpstream(defOp, targetBlock, visitedOps);
        }
      }
    }
  }
}

/// Performs a Depth-first Search (DFS) from a starting MLIR value to identify
/// all operations for duplication. It traverses the dataflow chain until it
/// either hits a store or a user-defined end operation. Operations that
/// must be duplicated are saved in `visitedOps`. The function
/// returns failure if it encounters a dead-end branch before reaching an end
/// operation, indicating an ill-defined duplication graph.
LogicalResult PipelineDuplicationPass::collectOpsDFS(
    mlir::Value currentVal, const std::vector<mlir::Operation *> &endOps,
    mlir::Block *currentBlock, llvm::DenseSet<mlir::Operation *> &visitedOps) {

  // if a value has no uses, it's a dead end branch without an end operation or
  // store, making the duplicated region ill-defined
  if (currentVal.use_empty()) {
    return failure();
  }

  // get the current block context to restrict our backward search and to check
  // if the dataflow has crossed into a different block
  mlir::Block *valBlock = currentVal.getParentBlock();
  if (valBlock != currentBlock) {
    // we changed block and need to make sure we have its terminator (branch)
    if (mlir::Operation *terminator = currentBlock->getTerminator()) {
      // returns true only if the item was newly inserted
      if (visitedOps.insert(terminator).second) {
        // collect all calculations in the block needed for the branch operands
        collectDependenciesUpstream(terminator, currentBlock, visitedOps);
      }
    }
  }

  // for each use of the value
  for (mlir::OpOperand &use : currentVal.getUses()) {
    // get the operation which uses the value
    mlir::Operation *user = use.getOwner();

    // check if already visited
    if (visitedOps.count(user)) {
      continue;
    }

    visitedOps.insert(user);
    // pull in its dependencies upstream which is necessary for
    // e.g. store operations where we need its address/index calculations
    collectDependenciesUpstream(user, valBlock, visitedOps);

    // an end operation or store operation is found which mark the end of the
    // duplicated region
    if (llvm::is_contained(endOps, user) || isa<mlir::memref::StoreOp>(user)) {
      continue;
    }

    // If we hit a terminator, we must follow the control flow paths
    if (user->hasTrait<mlir::OpTrait::IsTerminator>()) {
      /*
      if (auto branchOp = llvm::dyn_cast<mlir::BranchOpInterface>(user)) {
        mlir::Block *userBlock = user->getBlock();

        // Look through every successor block this branch can jump to
        for (unsigned i = 0; i < userBlock->getNumSuccessors(); ++i) {
          mlir::Block *successor = userBlock->getSuccessor(i);

          // Get the operands being forwarded to this specific successor
          auto successorOperands = branchOp.getSuccessorOperands(i);
          bool followedBlockArg = false;

          // Check if our current tracked value is one of the forwarded operands
          for (unsigned argIdx = 0; argIdx < successorOperands.size();
               ++argIdx) {
            if (successorOperands[argIdx] == currentVal) {
              // Map the operand to the corresponding Block Argument of the
              // successor block
              mlir::BlockArgument blockArg = successor->getArgument(argIdx);
              followedBlockArg = true;

              // Recurse into the successor block tracking the new block arg
              if (failed(
                      collectOpsDFS(blockArg, endOps, successor, visitedOps))) {
                return failure();
              }
            }
          }

          // Fallback if there are no block args
          if (!followedBlockArg && !successor->empty()) {
            mlir::Operation *firstOp = &successor->front();
            if (!visitedOps.count(firstOp)) {
              visitedOps.insert(firstOp);
              collectDependenciesUpstream(firstOp, successor, visitedOps);

              if (llvm::is_contained(endOps, firstOp) ||
                  isa<mlir::memref::StoreOp>(firstOp)) {
                continue;
              }

              // Use the first value (result) of the first operation to keep DFS
              // moving
              if (firstOp->getNumResults() > 0) {
                mlir::Value firstResult = firstOp->getResult(0);
                if (failed(collectOpsDFS(firstResult, endOps, successor,
                                         visitedOps))) {
                  return failure();
                }
              }
            }
          }
        }
      }
        */
      continue;
    }

    // if the operation has no results and is neither and end operation nor
    // a store, this makes the duplicated region ill-defined
    if (user->getNumResults() == 0) {
      return failure();
    }

    // recursive dfs step
    for (mlir::Value result : user->getResults()) {
      if (failed(collectOpsDFS(result, endOps, valBlock, visitedOps))) {
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
      // try to parse the string value of token as a base 10 int and store
      // in intVal. returns true if failed, false if succeeded
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

/// Parses and removes the `dynamatic.prediction_marker` operations from the
/// IR to populate `pragmaData`. It extracts the necessary data from each
/// marker's attributes and identifies the exact start and end operations
/// for each marker before erasing the physical markers from the function.
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

    // forward the marker's input directly to its users, making it possible
    // to safely delete the markers in the end
    op->getResult(0).replaceAllUsesWith(op->getOperand(0));
    markers.push_back(op);
  }

  // process the dynamatic.predict markers
  for (auto op : markers) {

    // extract the dictionary attribute 'dynamatic.predict' which contains
    // the pragma configuration details
    auto predictAttr =
        op->getAttrOfType<mlir::DictionaryAttr>("dynamatic.predict");
    if (!predictAttr) {
      llvm::errs()
          << "Error: This prediction marker doesn't have a "
             "dynamatic.predict "
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

      // verify attributes values and type were successfully found and
      // parsed for a new prediction marker ID. They must exist because the
      // start marker always has to come before any corresponding end
      // markers
      if (!valuesAttr || !typeAttr) {
        llvm::errs() << "Error: Malformed 'dynamatic.predict' values or type "
                        "attribute.\n";
        return failure();
      }

      // parse the list of values corresponding to the given data type
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

    // get the marker's operand which is the value it should replace
    mlir::Value markerInput = op->getOperand(0);
    llvm::StringRef location = locationAttr.getValue();

    // process the "start" marker to find the starting operation
    if (location == "start") {
      // a marker ID can only have one start operation
      if (markerMap[markerId].startOp != nullptr) {
        llvm::errs() << "Error: Marker ID " << markerId
                     << " has multiple start operations defined.\n";
        return failure();
      }
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

      // process the "end" marker to register the terminating operations of
      // the duplicated region
    } else if (location == "end") {
      // the endop is the definingop of the markerinput
      mlir::Operation *definingOp = markerInput.getDefiningOp();
      if (!definingOp) {
        llvm::errs() << "The end marker needs to be on an actual operation.\n";
        return failure();
      }
      markerMap[markerId].endOps.push_back(definingOp);
    }
  }

  // clean up the IR
  for (auto op : markers) {
    op->erase();
  }

  // populate pragmaData using the ordered markerId list
  for (auto &[op, data] : markerMap) {
    // ensure every collected prediction group has a defined entry point
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
  LLVM_DEBUG(llvm::errs() << "start reading predict marker data \n");
  if (failed(readPredictMarker(modOp, pragmaData)))
    return signalPassFailure();
  LLVM_DEBUG(llvm::errs() << "done reading predict marker data \n");

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

  int pragmaNumber = 1;
  // iterate over all pragmas with a start
  for (auto data : pragmaData) {

    mlir::Operation *startOp = data.startOp;
    mlir::Value predictInput = data.predInput;
    mlir::Block *targetBlock = startOp->getBlock();

    mlir::func::FuncOp funcOp =
        cast<mlir::func::FuncOp>(targetBlock->getParentOp());
    Location loc = funcOp.getLoc();

    // restructure the blocks
    mlir::Block *originalRemainderBlock = targetBlock->splitBlock(startOp);

    // DFS starting from the result of startop to find all ops that have to
    // be duplicated
    llvm::DenseSet<mlir::Operation *> visitedOps;
    visitedOps.insert(startOp);

    // if the startOp is also in endOps we only have to duplicate this one
    // operation and no DFS is necessary
    if (data.endOps.empty() ||
        (!data.endOps.empty() && startOp != data.endOps[0])) {
      for (auto op : startOp->getResults()) {
        llvm::errs() << "start dfs \n";
        if (failed(collectOpsDFS(op, data.endOps, originalRemainderBlock,
                                 visitedOps))) {
          llvm::errs()
              << "Could not find a valid graph to duplicate. Are all the "
                 "endops placed correctly?\n";
          return signalPassFailure();
        }
      }
      LLVM_DEBUG(llvm::errs() << "dfs succeeded\n");
    }

    for (auto op : visitedOps) {
      op->print(llvm::errs());
      llvm::errs() << '\n';
    }

    // check whether the duplicated graph is well-defined and save the
    // blocks that have to be replaced in a set
    auto result = validateDuplicationStructure(visitedOps, data.endOps);
    if (failed(result)) {
      llvm::errs() << "The graph to duplicate has an invalid structure.\n";
      return signalPassFailure();
    }
    auto &involvedBlocks = result->first;
    auto &terminationBlocks = result->second;

    // iterate through the function to sort as the operations collected
    // during the DFS might not be in the correct order
    llvm::SmallVector<mlir::Operation *> opsToMove;
    for (mlir::Block *block : involvedBlocks) {
      for (mlir::Operation &blockOp : *block) {
        if (visitedOps.count(&blockOp)) {
          opsToMove.push_back(&blockOp);
        }
      }
    }

    // DenseMap to track outside drivers per block, which have to be given
    // to the next block as block args
    llvm::DenseMap<mlir::Block *, llvm::SmallVector<mlir::Value>>
        blockOutsideDrivers;
    for (mlir::Operation *user : data.endOps) {
      mlir::Block *parentBlock = user->getBlock();
      // the results of this operation are added to the specific block's
      // vector
      for (mlir::Value res : user->getResults()) {
        blockOutsideDrivers[parentBlock].push_back(res);
      }
    }

    // map to keep track of the newly created block arguments per original
    // block
    llvm::DenseMap<mlir::Block *, llvm::SmallVector<mlir::Value>>
        blockToArgsMap;

    for (mlir::Block *origBlock : terminationBlocks) {
      // if this block didn't produce any outside drivers, it needs no new
      // arguments
      if (!blockOutsideDrivers.count(origBlock))
        continue;

      for (mlir::Value origOut : blockOutsideDrivers[origBlock]) {
        // add a block argument matching the type of the outside driver
        mlir::Value arg =
            origBlock->addArgument(origOut.getType(), origOut.getLoc());
        blockToArgsMap[origBlock].push_back(arg);
      }
    }

    // create the last block that will contain the original operations that
    // are not cloned
    mlir::Block *lastPath;
    // if we only duplicate operations inside of one block we need a new
    // block for the original operations
    if (*terminationBlocks.begin() == originalRemainderBlock) {
      lastPath = funcOp.addBlock();
    } else {
      lastPath = originalRemainderBlock;
    }

    // keep track of the newly created blocks
    llvm::DenseSet<mlir::Block *> trueBlocksSet;

    // iterator for the new names
    int i = 1;
    // PREDICTED PATHS
    for (mlir::Attribute attr : data.values) {
      // create branch condition
      builder.setInsertionPointToEnd(targetBlock);

      Value branchCond, constantComp;
      // the type is automatically saved in the attribute
      // it will deduct on its own whether it should 64bit or 32bit numbers
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

      mlir::Block *trueEntryBlock = funcOp.addBlock(); // true path
      mlir::Block *nextElseBlock;                      // false path
      if (i < (int)data.values.size())
        nextElseBlock = funcOp.addBlock();
      else
        nextElseBlock = lastPath;
      trueBlocksSet.insert(trueEntryBlock);
      trueBlocksSet.insert(nextElseBlock);

      builder.create<mlir::cf::CondBranchOp>(loc, branchCond, trueEntryBlock,
                                             nextElseBlock);

      // move the new blocks to right after targetBlock
      auto &blockList = funcOp.getBody().getBlocks();
      blockList.splice(std::next(targetBlock->getIterator()), blockList,
                       trueEntryBlock->getIterator());
      blockList.splice(std::next(trueEntryBlock->getIterator()), blockList,
                       nextElseBlock->getIterator());

      // setup the IR mapper to keep track of the duplicated operations
      mlir::IRMapping mapper;
      mapper.map(predictInput, constantComp);

      // generate all duplicated blocks
      // map original block to the duplicated blocks
      llvm::DenseMap<mlir::Block *, mlir::Block *> blockMap;
      mlir::Block *firstOrigBlock = opsToMove.front()->getBlock();
      blockMap[firstOrigBlock] = trueEntryBlock;
      mapper.map(firstOrigBlock, trueEntryBlock);

      // generate the rest of the blocks
      for (mlir::Block *origBlock : involvedBlocks) {
        if (origBlock == firstOrigBlock)
          continue; // skip because first block is already handled

        mlir::Block *newBlock = funcOp.addBlock();
        trueBlocksSet.insert(newBlock);

        // replicate the block arguments
        for (mlir::Value arg : origBlock->getArguments()) {
          mlir::Value newArg =
              newBlock->addArgument(arg.getType(), arg.getLoc());
          mapper.map(arg, newArg);
        }

        blockMap[origBlock] = newBlock;
        mapper.map(origBlock, newBlock);

        // position the block correctly
        blockList.splice(nextElseBlock->getIterator(), blockList,
                         newBlock->getIterator());
      }

      // perform the actual cloning
      for (mlir::Block *origBlock : involvedBlocks) {
        mlir::Block *clonedBlock = blockMap[origBlock];
        builder.setInsertionPointToEnd(clonedBlock);

        // clone all operations in this block
        for (mlir::Operation *origOp : opsToMove) {
          if (origOp->getBlock() != origBlock) {
            continue; // skip operations belonging to other blocks
          }

          mlir::Operation *cloned = builder.clone(*origOp, mapper);
          std::string newName = namer.getName(origOp).str() + "_dup" +
                                std::to_string(i * pragmaNumber);
          llvm::errs() << "newName: " << newName << '\n';
          cloned->setAttr("handshake.name", builder.getStringAttr(newName));

          for (unsigned j = 0; j < cloned->getNumResults(); j++) {
            // make sure cloned operations go to cloned operations
            mapper.map(origOp->getResult(j), cloned->getResult(j));
          }
        }

        // if it's an exit block, only forward the values produced in this
        // block
        if (terminationBlocks.count(origBlock)) {
          llvm::SmallVector<Value> trueBranchOperands;

          // check if this specific block even generates any outside drivers
          if (blockOutsideDrivers.count(origBlock)) {
            for (Value origOut : blockOutsideDrivers[origBlock]) {
              // map the original value to its newly cloned counterpart
              trueBranchOperands.push_back(mapper.lookup(origOut));
            }
          }

          // branch back to the original block, passing only its specific
          // exit arguments
          builder.create<mlir::cf::BranchOp>(loc, origBlock,
                                             trueBranchOperands);
        }
      }

      // Print Value mappings (Original Value -> Cloned Value)
      LLVM_DEBUG(llvm::errs() << "--- Mapper Contents ---\n");
      for (auto &pair : mapper.getValueMap()) {
        mlir::Value original = pair.first;
        mlir::Value cloned = pair.second;

        LLVM_DEBUG(llvm::errs() << "Value Mapping:\n");
        LLVM_DEBUG(llvm::errs() << "  From: " << original << "\n");
        LLVM_DEBUG(llvm::errs() << "  To:   " << cloned << "\n");
      }
      LLVM_DEBUG(llvm::errs() << "-----------------------\n");

      // update
      targetBlock = nextElseBlock;
      i++;
    }

    // Last path with the original logic
    // keep track of all newly create fallback blocks
    llvm::DenseSet<mlir::Block *> fallbackBlocks;

    for (mlir::Block *origTermBlock : terminationBlocks) {
      // create a new block for the last path if necessary
      mlir::Block *fallbackBlock;
      if (*terminationBlocks.begin() == originalRemainderBlock) {
        fallbackBlock = lastPath;
      } else {
        fallbackBlock = funcOp.addBlock();
      }
      fallbackBlocks.insert(fallbackBlock);

      // move the matching operations out of the original block
      for (mlir::Operation *origOp : opsToMove) {
        if (origOp->getBlock() == origTermBlock) {
          origOp->moveBefore(fallbackBlock, fallbackBlock->end());
        }
      }

      builder.setInsertionPointToEnd(fallbackBlock);
      builder.create<mlir::cf::BranchOp>(
          loc, origTermBlock, blockOutsideDrivers.lookup(origTermBlock));

      // Position it neatly right before the remainder block
      auto &blockList = funcOp.getBody().getBlocks();
      blockList.splice(origTermBlock->getIterator(), blockList,
                       fallbackBlock->getIterator());
    }

    // rewrite downstream users to read from the new block arguments
    for (auto &mapEntry : blockOutsideDrivers) {
      mlir::Block *origBlock = mapEntry.first;
      llvm::SmallVector<mlir::Value> &origOutputs = mapEntry.second;
      llvm::SmallVector<mlir::Value> &correspondingArgs =
          blockToArgsMap[origBlock];

      for (size_t it = 0; it < origOutputs.size(); it++) {
        mlir::Value origOut = origOutputs[it];
        mlir::Value newBlockArg = correspondingArgs[it];

        // Loop over your vector of structs to update the struct data itself
        for (PredictionData &data : pragmaData) {
          if (data.predInput == origOut) {
            data.predInput = newBlockArg;
          }
        }

        origOut.replaceUsesWithIf(newBlockArg, [&](OpOperand &operand) {
          mlir::Block *userBlock = operand.getOwner()->getBlock();
          // if the user is inside the original block itself, it should now
          // read from the block argument
          if (userBlock == origBlock) {
            return true;
          }

          // do not replace uses inside the cloned structures we generated,
          // the newly generated fallback blocks as well as the
          // involvedBlocks that are not terminatingblocks (this is checked
          // earlier)
          if (trueBlocksSet.count(userBlock) ||
              fallbackBlocks.count(userBlock) ||
              involvedBlocks.count(userBlock)) {
            return false;
          }

          // otherwise replace it if it's a downstream user outside our
          // target block within the same function scope
          return userBlock != targetBlock && userBlock->getParentOp() == funcOp;
        });
      }
    }
    pragmaNumber++;
  }

  // Automatically run push-constants pass after duplication
  mlir::PassManager pm(ctx);
  pm.addPass(dynamatic::createPushConstants());
  if (failed(pm.run(modOp)))
    return signalPassFailure();

  // dump the entire IR
  /* llvm::errs() << "=== IR BEFORE PUSH CONSTANTS AND HANDSHAKE ===\n";
  modOp.dump();
  llvm::errs() << "==============================================\n"; */
}
