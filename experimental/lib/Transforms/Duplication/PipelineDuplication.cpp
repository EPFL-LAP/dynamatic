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
  LogicalResult readFromJSON(const std::string &jsonPath, std::string &opName,
                             std::string &basicBlock, std::string &dataType,
                             std::vector<float> *floatList,
                             std::vector<int> *intList,
                             std::vector<double> *doubleList);

  void collectOpsDFS(mlir::Value currentVal,
                     llvm::DenseSet<mlir::Operation *> &sliceOps,
                     llvm::SmallVector<mlir::Value> &originalOutputs);

  template <typename T>
  std::pair<mlir::Value, mlir::Value>
  createBranchCond(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value selectRes, T comparisonValue);
};

} // namespace

LogicalResult PipelineDuplicationPass::readFromJSON(
    const std::string &jsonPath, std::string &opName, std::string &basicBlock,
    std::string &dataType, std::vector<float> *floatList,
    std::vector<int> *intList, std::vector<double> *doubleList) {

  // Open the .json file
  std::ifstream inputFile(jsonPath);
  if (!inputFile.is_open()) {
    llvm::errs() << "Failed to open unit positions file\n";
    return failure();
  }
  // Read the JSON content from the file and into a string
  std::string jsonString, line;
  while (std::getline(inputFile, line))
    jsonString += line;

  // Try to parse the string as a JSON
  llvm::Expected<llvm::json::Value> value = llvm::json::parse(jsonString);
  if (!value) {
    llvm::errs() << "Failed to parse unit positions in " << jsonPath << '\n';
    return failure();
  }

  const llvm::json::Object *rootObj = value->getAsObject();
  if (!rootObj)
    return failure();

  // Extract splitoperation -> operation-name
  if (const llvm::json::Object *splitOpObj =
          rootObj->getObject("splitoperation")) {
    if (auto parsedOpName = splitOpObj->getString("operation-name")) {
      opName = parsedOpName->str();
    }
  }

  if (auto parsedDataType = rootObj->getString("dataType")) {
    dataType = parsedDataType->str();
  } else
    return failure();

  if (auto parsedDataType = rootObj->getString("basicBlock")) {
    basicBlock = parsedDataType->str();
  } else
    return failure();

  const llvm::json::Array *parsedArray = rootObj->getArray("compValList");
  if (!parsedArray) {
    llvm::errs() << "Empty compValList not allowed.\n";
    return failure();
  }

  if (dataType == "float") {
    for (const llvm::json::Value &element : *parsedArray) {
      if (auto num = element.getAsNumber()) {
        floatList->push_back(static_cast<float>(*num));
      }
    }
  } else if (dataType == "double") {
    for (const llvm::json::Value &element : *parsedArray) {
      if (auto num = element.getAsNumber()) {
        floatList->push_back(*num);
      }
    }
  } else if (dataType == "integer" || dataType == "bool" ||
             dataType == "boolean" || dataType == "int") {
    for (const llvm::json::Value &element : *parsedArray) {
      if (auto num = element.getAsInteger()) {
        intList->push_back(static_cast<int>(*num));
      } else if (auto boolean = element.getAsBoolean()) {
        intList->push_back(*boolean ? 1 : 0);
      }
    }
    dataType = "int";
  } else {
    llvm::errs()
        << "Mismatch between JSON dataType and provided C++ vectors.\n";
    return failure();
  }

  return success();
}

void PipelineDuplicationPass::collectOpsDFS(
    mlir::Value currentVal, llvm::DenseSet<mlir::Operation *> &sliceOps,
    llvm::SmallVector<mlir::Value> &originalOutputs) {

  // if the current value has no users, it must be a leaf
  if (currentVal.use_empty()) {
    originalOutputs.push_back(currentVal);
  }

  for (mlir::OpOperand &use : currentVal.getUses()) {
    mlir::Operation *user = use.getOwner();

    // stop traversing if we hit memory operations or branches
    // TODO: replace with bb selection
    if (mlir::isa<mlir::memref::StoreOp, mlir::memref::LoadOp,
                  mlir::BranchOpInterface, mlir::func::ReturnOp>(user)) {

      if (originalOutputs.empty() || originalOutputs.back() != currentVal) {
        originalOutputs.push_back(currentVal);
      }
      continue;
    }

    // attempt to insert into the DenseSet
    // if (!visited)
    if (sliceOps.insert(user).second) {
      // Recursively traverse all outputs produced by this operation
      for (mlir::Value result : user->getResults()) {
        collectOpsDFS(result, sliceOps, originalOutputs);
      }
    }
  }
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

    branchCond = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, selectRes, constantComp);
  }

  return {branchCond, constantComp};
}

void PipelineDuplicationPass::runDynamaticPass() {
  mlir::ModuleOp modOp = getOperation();
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  // find operation
  // TODO: what is still missing?
  std::vector<float> floatValList;
  std::vector<int> intValList;
  std::vector<double> doubleValList;
  std::string opName;
  std::string dataType;
  std::string basicBlock;
  if (failed(PipelineDuplicationPass::readFromJSON(
          this->jsonPath, opName, basicBlock, dataType, &floatValList,
          &intValList, &doubleValList)))
    return signalPassFailure();

  llvm::errs() << "basic block: " << basicBlock << '\n';

  NameAnalysis &namer = getAnalysis<NameAnalysis>();
  Operation *op = namer.getOp(opName);
  if (!op) {
    llvm::errs() << "No operation named " << opName << " exists\n";
    return signalPassFailure();
  }
  mlir::Block *currentBlock = op->getBlock();
  mlir::func::FuncOp funcOp =
      cast<mlir::func::FuncOp>(currentBlock->getParentOp());
  Location loc = op->getLoc();
  Value selectRes = op->getResult(0);

  // restructure the blocks
  builder.setInsertionPointAfter(op);
  mlir::Block *exitBlock =
      currentBlock->splitBlock(builder.getInsertionPoint());
  // Block::iterator(branchCond.getDefiningOp())->getNextNode());

  // DFS starting from selectRes
  llvm::DenseSet<mlir::Operation *> unorganizedOps;
  llvm::SmallVector<mlir::Value> originalOutputs;
  collectOpsDFS(selectRes, unorganizedOps, originalOutputs);

  // iterate through the exitBlock to sort
  llvm::SmallVector<mlir::Operation *> opsToMove;
  for (mlir::Operation &blockOp : exitBlock->getOperations()) {
    if (unorganizedOps.count(&blockOp)) {
      opsToMove.push_back(&blockOp);
    }
  }

  llvm::SmallVector<Value> exitBlockArgs;
  for (Value origOut : originalOutputs) {
    exitBlockArgs.push_back(exitBlock->addArgument(origOut.getType(), loc));
  }

  // PREDICTED PATHS
  int size =
      std::max({floatValList.size(), intValList.size(), doubleValList.size()});
  for (int i = 0; i < size; i++) {
    // create branch condition
    builder.setInsertionPointToEnd(currentBlock);
    // assume only the first result is relevant
    Value branchCond, constantComp;
    if (dataType == "float" || dataType == "double") {
      std::tie(branchCond, constantComp) =
          createBranchCond(builder, loc, selectRes, floatValList[i]);
    } else if (dataType == "int") {
      std::tie(branchCond, constantComp) =
          createBranchCond(builder, loc, selectRes, intValList[i]);
    } else {
      llvm::errs() << "Error: Unknown or unsupported data type: " << dataType
                   << "\n";
      return signalPassFailure();
    }
    mlir::Block *trueBlock = funcOp.addBlock();     // true path
    mlir::Block *nextElseBlock = funcOp.addBlock(); // false path
    builder.create<mlir::cf::CondBranchOp>(loc, branchCond, trueBlock,
                                           nextElseBlock);

    // move the new blocks to right after currentBlock
    auto &blockList = funcOp.getBody().getBlocks();
    blockList.splice(std::next(currentBlock->getIterator()), blockList,
                     trueBlock->getIterator());
    blockList.splice(std::next(trueBlock->getIterator()), blockList,
                     nextElseBlock->getIterator());

    // clone the necessary operations to here
    builder.setInsertionPointToStart(trueBlock);
    mlir::IRMapping mapper;
    mapper.map(selectRes, constantComp);

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
    currentBlock = nextElseBlock;
  }

  // FALSE PATH
  // move all of the stuff from above here
  builder.setInsertionPointToStart(currentBlock);
  for (Operation *origOp : opsToMove) {
    origOp->moveBefore(currentBlock, currentBlock->end());
  }
  builder.setInsertionPointToEnd(currentBlock);
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

          return userBlock != currentBlock &&
                 userBlock->getParentOp() == funcOp && isCondBranch;
        });
  }

  // Automatically run push-constants pass after duplication
  mlir::PassManager pm(ctx);
  pm.addPass(dynamatic::createPushConstants());
  if (failed(pm.run(modOp)))
    return signalPassFailure();
}
