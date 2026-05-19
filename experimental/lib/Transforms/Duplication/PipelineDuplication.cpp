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
  LogicalResult readFromJSON(const std::string &jsonPath, float &compVal,
                             std::vector<float> &compValList,
                             std::string &opName);

  void collectOpsDFS(mlir::Value currentVal,
                     llvm::DenseSet<mlir::Operation *> &sliceOps);
};

} // namespace

LogicalResult PipelineDuplicationPass::readFromJSON(
    const std::string &jsonPath, float &compVal,
    std::vector<float> &compValList, std::string &opName) {

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

  // extract compVal, you can write 7 or 7.0
  if (auto parsedCompVal = rootObj->getInteger("compVal")) {
    compVal = static_cast<float>(*parsedCompVal);
  } else if (auto parsedCompValNum = rootObj->getNumber("compVal")) {
    compVal = static_cast<float>(*parsedCompValNum);
  }

  // Extract splitoperation -> operation-name
  if (const llvm::json::Object *splitOpObj =
          rootObj->getObject("splitoperation")) {
    if (auto parsedOpName = splitOpObj->getString("operation-name")) {
      opName = parsedOpName->str();
    }
  }

  if (const llvm::json::Array *parsedArray = rootObj->getArray("compValList")) {
    for (const llvm::json::Value &element : *parsedArray) {
      if (auto num = element.getAsNumber()) {
        compValList.push_back(static_cast<float>(*num));
      }
    }
  }

  return success();
}

void PipelineDuplicationPass::collectOpsDFS(
    mlir::Value currentVal, llvm::DenseSet<mlir::Operation *> &sliceOps) {
  for (mlir::OpOperand &use : currentVal.getUses()) {
    mlir::Operation *user = use.getOwner();

    // stop traversing if we hit memory operations or branches
    // TODO: what else??
    if (mlir::isa<mlir::memref::StoreOp, mlir::memref::LoadOp,
                  mlir::BranchOpInterface, mlir::func::ReturnOp>(user)) {
      continue;
    }

    // attempt to insert into the DenseSet
    // if (!visited)
    if (sliceOps.insert(user).second) {
      // Recursively traverse all outputs produced by this operation
      for (mlir::Value result : user->getResults()) {
        collectOpsDFS(result, sliceOps);
      }
    }
  }
}

void PipelineDuplicationPass::runDynamaticPass() {
  mlir::ModuleOp modOp = getOperation();
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  // Find select0 operation
  float compVal;
  std::vector<float> compValList;
  std::string opName;
  if (failed(PipelineDuplicationPass::readFromJSON(this->jsonPath, compVal,
                                                   compValList, opName)))
    return signalPassFailure();

  NameAnalysis &namer = getAnalysis<NameAnalysis>();
  Operation *op = namer.getOp(opName);
  if (!op) {
    llvm::errs() << "No operation named " << opName << " exists\n";
    return signalPassFailure();
  }
  mlir::Block *currentBlock = op->getBlock();
  mlir::func::FuncOp funcOp =
      dyn_cast<mlir::func::FuncOp>(currentBlock->getParentOp());
  Location loc = op->getLoc();
  Value selectRes = op->getResult(0);

  // restructure the blocks
  builder.setInsertionPointAfter(op);
  mlir::Block *exitBlock =
      currentBlock->splitBlock(builder.getInsertionPoint());
  // Block::iterator(branchCond.getDefiningOp())->getNextNode());

  // is there a better way than DFS + sorting?
  // DFS starting from selectRes
  llvm::DenseSet<mlir::Operation *> unorganizedOps;
  collectOpsDFS(selectRes, unorganizedOps);

  // iterate through the exitBlock to sort
  llvm::SmallVector<mlir::Operation *> opsToMove;
  for (mlir::Operation &blockOp : exitBlock->getOperations()) {
    if (unorganizedOps.count(&blockOp)) {
      opsToMove.push_back(&blockOp);
    }
  }

  // track the connecting to output wires
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
  }

  int counter = 2;
  for (float comp : compValList) {
    // create branch condition
    builder.setInsertionPointToEnd(currentBlock);
    // assume only the first result is relevant
    Value constantComp = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getFloatAttr(builder.getF32Type(), comp));
    Value branchCond = builder.create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::OEQ, selectRes, constantComp);
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
          namer.getName(origOp).str() + "_dup" + std::to_string(counter);
      cloned->setAttr("handshake.name", builder.getStringAttr(newName));

      for (unsigned i = 0; i < cloned->getNumResults(); ++i) {
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
    counter++;
  }

  // FALSE PATH
  // move all of the stuff from above here
  builder.setInsertionPointToStart(currentBlock);
  for (Operation *origOp : opsToMove) {
    origOp->moveBefore(currentBlock, currentBlock->end());
  }
  builder.setInsertionPointToEnd(currentBlock);
  builder.create<mlir::cf::BranchOp>(loc, exitBlock, originalOutputs);

  for (size_t i = 0; i < originalOutputs.size(); ++i) {
    originalOutputs[i].replaceUsesWithIf(
        exitBlockArgs[i], [&](OpOperand &operand) {
          mlir::Block *userBlock = operand.getOwner()->getBlock();
          if (userBlock == exitBlock) {
            return true;
          }

          // Otherwise, only replace it if it's completely outside the
          // true/false cascade structures we generated.
          return userBlock != currentBlock &&
                 userBlock->getParentOp() == funcOp &&
                 !llvm::any_of(compValList, [&](float) {
                   // Keeps replacements from corrupting the internal lanes
                   return userBlock->getTerminator() &&
                          isa<mlir::cf::CondBranchOp>(
                              userBlock->getTerminator());
                 });
        });
  }

  // Automatically run push-constants pass after duplication
  mlir::PassManager pm(ctx);
  pm.addPass(dynamatic::createPushConstants());
  if (failed(pm.run(modOp)))
    return signalPassFailure();
}
