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

  void runDynamaticPass() override;

private:
  LogicalResult readFromAttribute(mlir::ModuleOp modOp, mlir::Operation *&op,
                                  std::vector<int> &intValList,
                                  std::vector<mlir::Operation *> &endOps);

  LogicalResult collectOpsDFS(mlir::Value currentVal,
                              std::vector<mlir::Operation *> endOps,
                              llvm::DenseSet<mlir::Operation *> &visitedOps);

  template <typename T>
  std::pair<mlir::Value, mlir::Value>
  createBranchCond(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value selectRes, T comparisonValue);

  struct PredictionData;
  LogicalResult parseValuesList(llvm::StringRef valuesStr,
                                std::vector<int> &intVals,
                                std::vector<float> &floatVals,
                                std::string &dataType);

  LogicalResult readPredictMarker(mlir::ModuleOp modOp,
                                  std::vector<PredictionData> &pragmaData);
};

} // namespace

LogicalResult PipelineDuplicationPass::readFromAttribute(
    mlir::ModuleOp modOp, mlir::Operation *&op, std::vector<int> &intValList,
    std::vector<mlir::Operation *> &endOps) {
  int startCount = 0;

  // walk through all operations inside the module
  // iterate over the operations inside the function TODO:
  modOp.walk([&](mlir::Operation *currOp) {
    if (auto predictAttr =
            currOp->getAttrOfType<mlir::DictionaryAttr>("dynamatic.predict")) {
      auto locationAttr =
          predictAttr.get("location").dyn_cast_or_null<mlir::StringAttr>();
      if (!locationAttr)
        return;

      llvm::StringRef location = locationAttr.getValue();
      if (location == "start") {
        startCount++;
        op = currOp;

        // Extract the array of integers
        if (auto valuesAttr =
                predictAttr.get("values")
                    .dyn_cast_or_null<mlir::DenseI64ArrayAttr>()) {
          for (int64_t val : valuesAttr.asArrayRef()) {
            intValList.push_back(static_cast<int>(val));
          }
        }
      } else if (location == "end") {
        endOps.push_back(currOp);
      }
    }
  });

  // verification
  if (startCount != 1) {
    llvm::errs() << "Expected only one predict pragma with \"start\"\n";
    return failure();
  }
  if (endOps.empty()) {
    llvm::errs() << "An \"end\" predict pragma is necessary \n";
    return failure();
  }
  return success();
}

LogicalResult PipelineDuplicationPass::collectOpsDFS(
    mlir::Value currentVal, std::vector<mlir::Operation *> endOps,
    llvm::DenseSet<mlir::Operation *> &visitedOps) {

  // if a value has no uses, it's a dead end branch and neither a store nor an
  // endop
  if (currentVal.use_empty()) {
    return failure();
  }

  for (mlir::OpOperand &use : currentVal.getUses()) {
    mlir::Operation *user = use.getOwner();

    // check if already visited and mark
    if (visitedOps.count(user)) {
      continue;
    }
    visitedOps.insert(user);

    // endOp is found
    if (llvm::is_contained(endOps, user) || isa<mlir::memref::StoreOp>(user)) {
      continue;
    }

    // if the operation has no results (but this is handled above already?)
    if (user->getNumResults() == 0) {
      return failure();
    }

    // recursive step
    for (mlir::Value result : user->getResults()) {
      if (failed(collectOpsDFS(result, endOps, visitedOps))) {
        return failure();
      }
    }
  }
  return success();
}

template <typename T>
std::pair<mlir::Value, mlir::Value> PipelineDuplicationPass::createBranchCond(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value selectRes,
    T comparisonValue) {

  Value constantComp;
  Value branchCond;
  if constexpr (std::is_floating_point_v<T>) {
    // float or double
    mlir::Type floatTy =
        std::is_same_v<T, double> ? builder.getF64Type() : builder.getF32Type();

    constantComp = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getFloatAttr(floatTy, comparisonValue));

    branchCond = builder.create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::OEQ, selectRes, constantComp);

  } else if constexpr (std::is_integral_v<T>) {
    // int or bool
    constantComp = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getIntegerAttr(builder.getI32Type(), comparisonValue));
    llvm::errs() << "building of constantComp succeeded: " << constantComp
                 << '\n';

    branchCond = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, selectRes, constantComp);
    llvm::errs() << "building of branchCond succeeded: " << branchCond << '\n';
  } else {
    llvm::errs() << "this should not happen!\n";
  }

  return {branchCond, constantComp};
}

struct PipelineDuplicationPass::PredictionData {
  mlir::Operation *startOp;
  mlir::Value predInput;
  std::vector<mlir::Operation *> endOps;
  std::vector<float> floatValList;
  std::vector<int> intValList;
  std::string dataType;
};

// TODO: add strings and maybe more types
LogicalResult PipelineDuplicationPass::parseValuesList(
    llvm::StringRef valuesStr, std::vector<int> &intVals,
    std::vector<float> &floatVals, std::string &dataType) {

  std::string s = valuesStr.str();

  if (!s.empty() && s.front() == '[')
    s.erase(0, 1);
  if (!s.empty() && s.back() == ']')
    s.pop_back();
  if (s.empty())
    return failure();

  if (s.find('.') != std::string::npos) {
    dataType = "float";
    double doubleVal;
    if (llvm::StringRef(s).getAsDouble(doubleVal)) {
      return failure();
    }
    floatVals.push_back(static_cast<float>(doubleVal));
    return success();

  } else {
    dataType = "int";
    int intVal;
    if (llvm::StringRef(s).getAsInteger(10, intVal)) {
      return failure();
    }
    intVals.push_back(intVal);
    return success();
  }
}

LogicalResult PipelineDuplicationPass::readPredictMarker(
    mlir::ModuleOp modOp, std::vector<PredictionData> &pragmaData) {

  std::map<int, PredictionData> markerMap;
  std::vector<mlir::Operation *> markersToErase;

  auto funcOps = modOp.getOps<mlir::func::FuncOp>();
  mlir::func::FuncOp funcOp = *funcOps.begin();

  for (mlir::Operation &opRef : funcOp.getOps()) {
    mlir::Operation *op = &opRef;
    if (op->getName().getStringRef() != "dynamatic.prediction_marker")
      continue;
    llvm::errs() << "found a prediction marker\n";
    // extract the dictionary attribute 'dynamatic.predict'
    auto predictAttr =
        op->getAttrOfType<mlir::DictionaryAttr>("dynamatic.predict");
    if (!predictAttr) {
      llvm::errs() << "no dynamatic.predict attribute!\n";
      return failure();
    }

    auto locationAttr = predictAttr.getAs<mlir::StringAttr>("location");
    auto markerAttr = predictAttr.getAs<mlir::IntegerAttr>("marker");
    auto valuesAttr = predictAttr.getAs<mlir::StringAttr>("values");

    int markerId = markerAttr.getInt();
    llvm::StringRef location = locationAttr.getValue();

    // initialize the struct for this marker ID if it doesn't yet exist
    if (markerMap.find(markerId) == markerMap.end()) {
      PredictionData newData;
      newData.startOp = nullptr;
      if (valuesAttr) {
        if (failed(parseValuesList(valuesAttr.getValue(), newData.intValList,
                                   newData.floatValList, newData.dataType))) {
          llvm::errs() << "Failed to parse value attributes for marker ID "
                       << markerId << '\n';
          return failure();
        }
      } else
        return failure();
      markerMap[markerId] = newData;
    }
    mlir::Value markerInput = op->getOperand(0);
    mlir::Value markerResult = op->getResult(0);
    if (location == "start") {
      mlir::Operation *nextNode = op->getNextNode();
      bool isUser = false;
      for (mlir::Value operand : nextNode->getOperands()) {
        if (operand == markerResult) {
          isUser = true;
          break;
        }
      }
      if (isUser) {
        markerMap[markerId].startOp = nextNode;
      } else {
        llvm::errs() << "Warning: Next node does not use the marker\n";
        markerMap[markerId].startOp = *op->getResult(0).user_begin();
      }
      markerMap[markerId].predInput = markerInput;

    } else if (location == "end") {
      // The endOp is the operation defining the marker's operand (arith.addi ->
      // %11)
      mlir::Operation *definingOp = markerInput.getDefiningOp();
      markerMap[markerId].endOps.push_back(definingOp);
    }

    markersToErase.push_back(op);
  }

  for (auto op : markersToErase) {
    op->getResult(0).replaceAllUsesWith(op->getOperand(0));
    op->erase();
  }

  for (auto &pair : markerMap) {
    PredictionData &data = pair.second;
    if (data.endOps.empty()) {
      llvm::errs() << "Marker ID " << pair.first
                   << "does not have any endoperations\n";
      return failure();
    }

    if (!data.startOp) {
      llvm::errs() << "Marker ID " << pair.first
                   << "does not have a valid start operation\n";
      return failure();
    }
    pragmaData.push_back(std::move(pair.second));
  }
  return success();
}

void PipelineDuplicationPass::runDynamaticPass() {
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
  for (auto &data : pragmaData) {
    llvm::errs() << "Data Type : " << data.dataType << "\n";

    // Print values list based on data type
    llvm::errs() << "Values    : [";
    if (data.dataType == "float") {
      for (size_t i = 0; i < data.floatValList.size(); ++i) {
        llvm::errs() << data.floatValList[i]
                     << (i + 1 < data.floatValList.size() ? ", " : "");
      }
    } else {
      for (size_t i = 0; i < data.intValList.size(); ++i) {
        llvm::errs() << data.intValList[i]
                     << (i + 1 < data.intValList.size() ? ", " : "");
      }
    }
    llvm::errs() << "]\n";

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

  mlir::Operation *startOp = pragmaData[0].startOp;
  mlir::Value predictInput = pragmaData[0].predInput;
  mlir::Block *targetBlock = startOp->getBlock();

  mlir::func::FuncOp funcOp =
      cast<mlir::func::FuncOp>(targetBlock->getParentOp());
  Location loc = funcOp.getLoc();

  // restructure the blocks
  // builder.setInsertionPointAfter(startOp);
  mlir::Block *exitBlock = targetBlock->splitBlock(startOp);
  mlir::Block *falseBlock = funcOp.addBlock();

  // DFS starting from predictInput to find all ops that have to be duplicated
  llvm::DenseSet<mlir::Operation *> visitedOps;
  visitedOps.insert(startOp);
  if (startOp->getResult(0) == 0 ||
      failed(collectOpsDFS(startOp->getResult(0), pragmaData[0].endOps,
                           visitedOps))) {
    llvm::errs() << "Could not find a valid graph to duplicate. Are all the "
                    "endops placed correctly?\n";
    return signalPassFailure();
  }

  llvm::errs() << "dfs succeeded\n";

  // iterate through the exitBlock to sort
  llvm::SmallVector<mlir::Operation *> opsToMove;
  for (mlir::Operation &blockOp : funcOp.getOps()) {
    if (visitedOps.count(&blockOp)) {
      opsToMove.push_back(&blockOp);
      blockOp.dump();
    }
  }

  // identify which values are also used outside
  llvm::SmallVector<mlir::Value> originalOutputs;
  for (mlir::Operation *origOp : opsToMove) {
    for (mlir::Value origRes : origOp->getResults()) {
      // Determine if a value breaches the edge of our duplicated region
      bool usedOutside =
          llvm::any_of(origRes.getUsers(), [&](mlir::Operation *user) {
            return std::find(opsToMove.begin(), opsToMove.end(), user) ==
                   opsToMove.end();
          });

      if (usedOutside || origRes.use_empty()) {
        originalOutputs.push_back(origRes);
      }
    }
  }

  llvm::SmallVector<Value> exitBlockArgs;
  for (Value origOut : originalOutputs) {
    exitBlockArgs.push_back(exitBlock->addArgument(origOut.getType(), loc));
    llvm::errs() << origOut << " origOut\n ";
  }

  // PREDICTED PATHS
  int size = std::max(pragmaData[0].floatValList.size(),
                      pragmaData[0].intValList.size());
  for (int i = 0; i < size; i++) {
    // create branch condition
    builder.setInsertionPointToEnd(targetBlock);
    llvm::errs() << "start for loop \n";
    // assume only the first result is relevant
    Value branchCond, constantComp;
    if (pragmaData[0].dataType == "float" ||
        pragmaData[0].dataType == "double") {
      std::tie(branchCond, constantComp) = createBranchCond(
          builder, loc, predictInput, pragmaData[0].floatValList[i]);
    } else if (pragmaData[0].dataType == "int") {
      llvm::errs() << pragmaData[0].intValList[i] << ", " << predictInput
                   << '\n';
      std::tie(branchCond, constantComp) = createBranchCond(
          builder, loc, predictInput, pragmaData[0].intValList[i]);
      llvm::errs() << "branchCond: " << branchCond
                   << "\nconstantComp: " << constantComp << '\n';
    } else {
      llvm::errs() << "Error: Unknown or unsupported data type: "
                   << pragmaData[0].dataType << "\n";
      return signalPassFailure();
    }
    mlir::Block *trueBlock = funcOp.addBlock(); // true path
    mlir::Block *nextElseBlock;                 // false path
    if (i + 1 < size)
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

      for (unsigned i = 0; i < cloned->getNumResults(); ++i) {
        // make sure cloned operations go to cloned operations
        mapper.map(origOp->getResult(i), cloned->getResult(i));
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
    originalOutputs[i].replaceUsesWithIf(
        exitBlockArgs[i], [&](OpOperand &operand) {
          mlir::Block *userBlock = operand.getOwner()->getBlock();
          if (userBlock == exitBlock) {
            return true;
          }

          // Otherwise, only replace it if it's completely outside the
          // true/false cascade structures we generated.
          bool isCondBranch =
              userBlock->getTerminator() &&
              isa<mlir::cf::CondBranchOp>(userBlock->getTerminator());

          return userBlock != targetBlock &&
                 userBlock->getParentOp() == funcOp && isCondBranch;
        });
  }

  // Automatically run push-constants pass after duplication
  mlir::PassManager pm(ctx);
  pm.addPass(dynamatic::createPushConstants());
  if (failed(pm.run(modOp)))
    return signalPassFailure();
}
