// Include some other useful headers.
#include "dynamatic/Analysis/NameAnalysis.h" // needed
#include "dynamatic/Support/DynamaticPass.h"
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
                             std::string &opName);
};

} // namespace

// TODO: change compVal into list / integers?
LogicalResult PipelineDuplicationPass::readFromJSON(const std::string &jsonPath,
                                                    float &compVal,
                                                    std::string &opName) {

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

  return success();
}

void PipelineDuplicationPass::runDynamaticPass() {
  mlir::ModuleOp modOp = getOperation();
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  // Find select0 operation
  float compVal;
  std::string opName;
  if (failed(PipelineDuplicationPass::readFromJSON(this->jsonPath, compVal,
                                                   opName)))
    return signalPassFailure();

  NameAnalysis &namer = getAnalysis<NameAnalysis>();
  Operation *rawOp = namer.getOp(opName);
  if (!rawOp) {
    llvm::errs() << "No operation named " << opName << " exists\n";
    return signalPassFailure();
  }
  // TODO: make this mlir::arith::xxx changeable
  auto op = dyn_cast<mlir::arith::SelectOp>(rawOp);
  mlir::Block *currentBlock = op->getBlock();
  mlir::func::FuncOp funcOp =
      dyn_cast<mlir::func::FuncOp>(currentBlock->getParentOp());
  Location loc = op.getLoc();

  // create branch condition
  builder.setInsertionPointAfter(op);
  // TODO: get this info from the json file? or other
  Value selectRes = op.getResult();

  Value constantComp = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getFloatAttr(builder.getF32Type(), compVal));

  Value branchCond = builder.create<mlir::arith::CmpFOp>(
      loc, mlir::arith::CmpFPredicate::OEQ, selectRes, constantComp);

  // restructure the blocks
  // TODO: splitblock maybe works differently for other operations
  mlir::Block *exitBlock = currentBlock->splitBlock(
      Block::iterator(branchCond.getDefiningOp())->getNextNode());
  mlir::Block *trueBlock = funcOp.addBlock();  // true path
  mlir::Block *falseBlock = funcOp.addBlock(); // false path

  builder.setInsertionPointToEnd(currentBlock);
  builder.create<mlir::cf::CondBranchOp>(loc, branchCond, trueBlock,
                                         falseBlock);

  // TRUE PATH
  // clone the necessary operations to here as well as the constant
  builder.setInsertionPointToStart(trueBlock);
  Value newConstant = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getFloatAttr(builder.getF32Type(), compVal));

  mlir::IRMapping mapper;
  // we only care about values derived from our starting operation which in
  // this hardcoded case is the selectop
  llvm::DenseSet<Value> trackedValues;
  for (Value res : op->getResults()) {
    trackedValues.insert(res);
    mapper.map(res, newConstant);
  }

  llvm::SmallVector<Operation *, 4> opsToMove;
  Value lastClonedRes;
  for (Operation &blockOp : exitBlock->getOperations()) {
    // stop if we hit a store (or what else?)
    // TODO: how does this end?
    if (isa<mlir::memref::StoreOp, mlir::BranchOpInterface>(blockOp)) {
      break;
    }

    // check if the op uses a value we are tracking
    bool isDependent = llvm::any_of(blockOp.getOperands(), [&](Value operand) {
      return trackedValues.count(operand);
    });

    if (isDependent) {
      opsToMove.push_back(&blockOp);
      Operation *cloned = builder.clone(blockOp, mapper);
      llvm::StringRef originalName = namer.getName(&blockOp);
      std::string newName = originalName.str() + "_dup";
      cloned->setAttr("handshake.name",
                      builder.getStringAttr(originalName.str() + "_dup"));

      // track the result of the cloned operation if it produces one
      if (cloned->getNumResults() > 0) {
        lastClonedRes = cloned->getResult(0);
      }

      // track the new results so we can find the next operations in the chain
      for (auto it : llvm::enumerate(cloned->getResults())) {
        size_t index = it.index();
        Value clonedRes = it.value();
        Value originalRes = blockOp.getResult(index);

        // track the original result so we can find its users later in the
        // block
        trackedValues.insert(originalRes);
        mapper.map(originalRes, clonedRes);
      }
    }
  }

  llvm::errs() << "--- Mapper Contents ---\n";

  // Print Value mappings (Original Value -> Cloned Value)
  for (auto &pair : mapper.getValueMap()) {
    mlir::Value original = pair.first;
    mlir::Value cloned = pair.second;

    llvm::errs() << "Value Mapping:\n";
    llvm::errs() << "  From: " << original << "\n";
    llvm::errs() << "  To:   " << cloned << "\n";
  }

  // print mappings
  for (auto &pair : mapper.getBlockMap()) {
    mlir::Block *original = pair.first;
    mlir::Block *cloned = pair.second;

    llvm::errs() << "Block Mapping:\n";
    original->printAsOperand(llvm::errs());
    llvm::errs() << " -> ";
    cloned->printAsOperand(llvm::errs());
    llvm::errs() << "\n";
  }

  llvm::errs() << "-----------------------\n";

  builder.setInsertionPointToEnd(trueBlock);
  Value exitBlockArg = exitBlock->addArgument(lastClonedRes.getType(), loc);
  builder.create<mlir::cf::BranchOp>(loc, exitBlock, lastClonedRes);

  // FALSE PATH
  // move all of the stuff from above here
  builder.setInsertionPointToStart(falseBlock);
  for (Operation *origOp : opsToMove) {
    llvm::errs() << "Moving op: ";
    origOp->print(llvm::errs());
    llvm::errs() << "\n";
    origOp->moveBefore(falseBlock, falseBlock->end());
  }
  Value lastOrigRes = opsToMove.back()->getResult(0);
  // make sure in the new block it maps to the correct value
  lastOrigRes.replaceAllUsesWith(exitBlockArg);
  builder.create<mlir::cf::BranchOp>(loc, exitBlock, lastOrigRes);
}
