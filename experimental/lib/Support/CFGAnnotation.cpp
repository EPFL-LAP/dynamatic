//===- CFGAnnotation.cpp --- CFG Annotation ---------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements a set of utilites to annotate the CFG in an handshake function,
// parse the information back, re-build the cf structure and flatten it back.
// The following steps should be taken into account to use this library (related
// to the dependencies with the func and cf dialects)
//
// 1. When declaring the pass in `Passes.td`, add the following two dialects as
// dependencies: mlir::cf::ControlFlowDialect, mlir::func::FuncDialect";
// 2. In the header file of the same pass, add the following inclusions:
//    #include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
//    #include "mlir/Dialect/Func/IR/FuncOps.h"
// 3. In the CMake file of the same pass, add the following dependencies:
//    MLIRControlFlowDialect
//    MLIRFuncDialect
//
//===----------------------------------------------------------------------===//
//
#include "experimental/Support/CFGAnnotation.h"
#include "dynamatic/Support/CFG.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;

bool dynamatic::experimental::cfg::CFGNode::isConditional() const {
  return conditionName.has_value();
}

bool dynamatic::experimental::cfg::CFGNode::isUnconditional() const {
  return !conditionName.has_value();
}

unsigned dynamatic::experimental::cfg::CFGNode::getSuccessor() const {
  return isUnconditional() ? std::get<unsigned>(successors) : -1;
}
unsigned dynamatic::experimental::cfg::CFGNode::getTrueSuccessor() const {
  return isConditional()
             ? std::get<std::pair<unsigned, unsigned>>(successors).first
             : -1;
}
unsigned dynamatic::experimental::cfg::CFGNode::getFalseSuccessor() const {
  return isConditional()
             ? std::get<std::pair<unsigned, unsigned>>(successors).second
             : -1;
}
std::string dynamatic::experimental::cfg::CFGNode::getCondition() const {
  return isConditional() ? conditionName.value() : "";
}

/// Get a CFG from the annotation on an handshake function
static FailureOr<cfg::CFGAnnotation>
unserializeEdges(handshake::FuncOp &funcOp) {

  if (!funcOp->hasAttr(cfg::CFG_EDGES)) {
    return funcOp->emitError() << "Handshake function has no annotation `"
                               << cfg::CFG_EDGES << "`\n";
  }

  auto edges = funcOp->getAttrOfType<mlir::StringAttr>(cfg::CFG_EDGES).str();

  // Index of he current character in the string
  unsigned currentIndex = 0;
  // Since of the string
  unsigned stringSize = edges.size();
  // Result map
  cfg::CFGAnnotation result;

  // Function to predict the current character in the string. Returns false if
  // the character's different from what's expected. The index is also
  // incremented.
  auto expectCharacter = [&](char c) -> bool {
    return edges[currentIndex++] == c;
  };

  // Return true if the index has covered the whole string
  auto isStringDone = [&]() -> bool { return currentIndex >= stringSize; };

  // Lookup the current character in the string.
  auto lookUp = [&]() -> char { return edges[currentIndex]; };

  // Return a parsing error
  auto parseError = [&]() -> FailureOr<cfg::CFGAnnotation> {
    return funcOp->emitError()
           << "Fail while parsing `" << cfg::CFG_EDGES << "`\n";
  };

  // Parse a number in the string; all the character in the string are taken
  // until a non-digit character is found.
  auto parseNumber = [&]() -> int {
    unsigned sourceSize = 0;
    unsigned sourceStart = currentIndex;
    while (currentIndex < stringSize && std::isdigit(edges[currentIndex]))
      sourceSize++, currentIndex++;
    if (!sourceSize || isStringDone())
      return -1;

    return std::stoi(edges.substr(sourceStart, sourceSize));
  };

  // Parse a sub-string in the string; all the character in the string are taken
  // until a non-alpha numeric character is found.
  auto parseString = [&]() -> std::string {
    unsigned sourceSize = 0;
    unsigned sourceStart = currentIndex;
    while (currentIndex < stringSize && std::isalnum(edges[currentIndex]))
      sourceSize++, currentIndex++;
    if (!sourceSize || currentIndex >= stringSize)
      return "";

    return edges.substr(sourceStart, sourceSize);
  };

  // First character is an open curly bracket
  if (!expectCharacter(cfg::OPEN_LIST))
    return parseError();

  // Possibly the list is empty: in this case, there is no edge in the CFG, and
  // we return an empty list
  if (lookUp() == cfg::CLOSE_LIST && isStringDone())
    return result;

  // While there is an edge to parse
  while (currentIndex < stringSize) {

    // Peek an open square bracket
    if (!expectCharacter(cfg::OPEN_EDGE))
      return parseError();

    // Parse the source node
    int sourceEdge = parseNumber();
    if (sourceEdge < 0)
      return parseError();

    // Expect a delimiter to bre presetn
    if (!expectCharacter(cfg::DELIMITER))
      return parseError();

    // Parse the source node
    int destTrue = parseNumber();
    if (destTrue < 0)
      return parseError();

    // If there is a DELIMITER, then we have a conditional branch
    if (lookUp() == cfg::DELIMITER) {

      currentIndex++;

      // Parse the destination node
      int destFalse = parseNumber();
      if (destFalse < 0)
        return parseError();

      // Parse a delimiter
      if (!expectCharacter(cfg::DELIMITER))
        return parseError();

      // Parse the string representing the instruction providing the branch
      // condition
      std::string condition = parseString();
      if (condition == "")
        return parseError();

      // Create a conditional node
      result.insert({sourceEdge, cfg::CFGNode(destTrue, destFalse, condition)});

    } else if (lookUp() == cfg::CLOSE_EDGE) {
      // Create a non-conditional node
      result.insert({sourceEdge, cfg::CFGNode(destTrue)});
    } else {
      return parseError();
    }

    // Expect the end of an edge
    if (!expectCharacter(cfg::CLOSE_EDGE))
      return parseError();

    // Stop if we are at the end of the list
    if (lookUp() == cfg::CLOSE_LIST)
      break;
  }

  if (!expectCharacter(cfg::CLOSE_LIST) || currentIndex != stringSize)
    return parseError();

  return result;
}

/// Extract the CFG information from a region. Each basic block with an out
/// edge is marked with the successor basic blocks, related to the structure
/// `CFGEdge` for edges.
static cfg::CFGAnnotation getCFGEdges(Region &funcRegion) {

  cfg::CFGAnnotation edgeMap;

  // Get the ID of a block through the ID annotated in the terminator operation
  auto getIDBlock = [&](Block *bb) -> unsigned {
    auto idOptional = getLogicBB(bb->getTerminator());
    if (!idOptional.has_value())
      bb->getTerminator()->emitError() << "Operation has no BB annotated\n";
    return idOptional.value();
  };

  // For each block in the IR
  for (auto &block : funcRegion.getBlocks()) {

    // Get the terminator and its block ID
    auto *terminator = block.getTerminator();
    unsigned blockID = getIDBlock(&block);

    // If the terminator is a branch, then the edge is unconditional. If the
    // terminator is `cond_br`, then the branch is conditional.
    if (auto branchOp = dyn_cast<cf::BranchOp>(terminator); branchOp) {
      edgeMap.insert({blockID, dynamatic::experimental::cfg::CFGNode(
                                   getIDBlock(branchOp->getSuccessor(0)))});
    } else if (auto condBranchOp = dyn_cast<cf::CondBranchOp>(terminator);
               condBranchOp) {
      // Get the name of the operation which defines the condition used for the
      // branch
      Operation *conditionOperation =
          condBranchOp.getOperand(0).getDefiningOp();
      std::string conditionName =
          conditionOperation->getAttrOfType<mlir::StringAttr>("handshake.name")
              .str();

      // Get IDs of both true and false destinations
      unsigned trueDestID = getIDBlock(condBranchOp.getTrueDest());
      unsigned falseDestID = getIDBlock(condBranchOp.getFalseDest());
      edgeMap.insert(
          {blockID, cfg::CFGNode(trueDestID, falseDestID, conditionName)});
    } else if (!llvm::isa_and_nonnull<func::ReturnOp>(terminator)) {
      terminator->emitError()
          << "Not a cf terminator for BB" << blockID << "\n";
    }
  }

  return edgeMap;
}

void dynamatic::experimental::cfg::annotateCFG(
    handshake::FuncOp &funcOp, ConversionPatternRewriter &rewriter) {

  // Get the CFG information
  const auto edgeMap = getCFGEdges(funcOp.getRegion());

  std::string result = "";

  auto appendSymbol = [&](char s) { result += s; };
  auto appendStr = [&](const std::string &s) { result += s; };

  appendSymbol(OPEN_LIST);
  for (auto &[sourceBB, edge] : edgeMap) {
    appendSymbol(OPEN_EDGE);
    appendStr(std::to_string(sourceBB));
    appendSymbol(DELIMITER);
    if (edge.isConditional()) {
      appendStr(std::to_string(edge.getTrueSuccessor()));
      appendSymbol(DELIMITER);
      appendStr(std::to_string(edge.getFalseSuccessor()));
      appendSymbol(DELIMITER);
      appendStr(edge.getCondition());
    } else {
      appendStr(std::to_string(edge.getSuccessor()));
    }
    appendSymbol(CLOSE_EDGE);
  }
  appendSymbol(CLOSE_LIST);

  funcOp->setAttr(CFG_EDGES, rewriter.getStringAttr(result));
}

LogicalResult dynamatic::experimental::cfg::restoreCfStructure(
    handshake::FuncOp &funcOp, ConversionPatternRewriter &rewriter) {

  // Get an operation according to its name. If not present, return a compare
  // operation in the block provided as input
  auto getOpByName = [&](const std::string &name,
                         unsigned blockID) -> Operation * {
    constexpr llvm::StringLiteral nameAttr("handshake.name");
    Operation *cmpSameBlock = nullptr;

    // For each operaiton
    for (Operation &op : funcOp.getOps()) {

      // Continue if the operation has no name attribute
      if (!op.hasAttr(nameAttr))
        continue;

      // Save the operation if it is a compare (both integer and floating) in
      // the same block of the provdied one
      if (llvm::isa_and_nonnull<handshake::CmpIOp, handshake::CmpFOp>(op) &&
          getLogicBB(&op).value() == blockID)
        cmpSameBlock = &op;

      // Return the operation if it has the same name as the input
      auto opName = op.getAttrOfType<mlir::StringAttr>(nameAttr);
      std::string opNameStr = opName.str();
      if (name == opNameStr)
        return &op;
    }
    return cmpSameBlock;
  };

  // Get the map of edges
  auto failureOrEdges = unserializeEdges(funcOp);
  if (failed(failureOrEdges))
    return failure();
  CFGAnnotation &edges = *failureOrEdges;

  // Maintains the ID of the current block under analysis
  unsigned currentBlockID = 0;

  // Maintains the current block under analysis: all the operations in block 0
  // are maintained in the same location, while the others operations are
  // inserted in the new blocks
  Block *currentBlock = &funcOp.getBlocks().front();

  // Keep a mapping between each index and each block. Since the block 0 is kept
  // as it is, it can be inserted already in the map.
  DenseMap<unsigned, Block *> indexToBlock;
  indexToBlock.insert({0, currentBlock});

  // Temporary store all the operations in the original function
  SmallVector<Operation *> originalOps;
  for (auto &op : funcOp.getOps())
    originalOps.push_back(&op);

  // For each operation
  for (auto *op : originalOps) {

    // Get block ID of the current operation. If it is not annotated (end
    // operation/LSQ/memory operation) then use the current ID
    unsigned opBlock = getLogicBB(op).value_or(currentBlockID);

    // Do not modify the block structure if we are in block 0
    if (opBlock == 0)
      continue;

    // If a new ID is found with respect to the old one, then create a new block
    // in the function
    if (opBlock != currentBlockID) {
      currentBlock = funcOp.addBlock();
      indexToBlock.insert({++currentBlockID, currentBlock});
    }

    // Move the current operation at the end of the new block we are currently
    // using
    op->moveBefore(currentBlock, currentBlock->end());
  }

  // Once we are done creating the blocks, we need to insert the terminators to
  // obtain a proper block structure, using the edge information provided as
  // input

  // For each block
  for (auto [blockID, bb] : llvm::enumerate(funcOp.getBlocks())) {
    rewriter.setInsertionPointToEnd(&bb);

    if (!edges.contains(blockID)) {
      rewriter.create<func::ReturnOp>(bb.back().getLoc());
      continue;
    }

    auto edge = edges.lookup(blockID);

    // Either create a conditional or unconditional branch depending on the type
    // of edge we have
    if (edge.isConditional()) {
      Operation *condOp = getOpByName(edge.getCondition(), blockID);
      if (!condOp)
        return failure();
      rewriter.create<cf::CondBranchOp>(bb.back().getLoc(),
                                        condOp->getResult(0),
                                        indexToBlock[edge.getTrueSuccessor()],
                                        indexToBlock[edge.getFalseSuccessor()]);
    } else {
      unsigned successor = edge.getSuccessor();
      rewriter.create<cf::BranchOp>(bb.back().getLoc(),
                                    indexToBlock[successor]);
    }
  }

  return success();
}

LogicalResult
dynamatic::experimental::cfg::flattenFunction(handshake::FuncOp &funcOp) {

  // For each block, remove the terminator, which is supposed to be either a
  // cf::condBranchOp, cf::BranchOp or func::ReturnOp
  for (auto &bb : funcOp.getBlocks()) {
    Operation *op = &bb.back();
    if (!llvm::isa_and_nonnull<mlir::cf::CondBranchOp, mlir::cf::BranchOp,
                               mlir::func::ReturnOp>(op)) {
      return op->emitError("Last operation of the block must be "
                           "cf::CondBranchOp, cf::BranchOp or func::ReturnOp");
    }
    op->erase();
  }

  // Move each operation in the entry block
  for (Block &block : llvm::make_early_inc_range(llvm::drop_begin(funcOp))) {
    llvm::SmallVector<Operation *> ops;
    for (auto &op : block.getOperations())
      ops.push_back(&op);
    for (Operation *op : ops)
      op->moveAfter(&funcOp.front().back());
  }

  // Remove all the non initial blcok
  for (Block &block : llvm::make_early_inc_range(llvm::drop_begin(funcOp))) {
    block.dropAllUses();
    block.erase();
  }

  assert(funcOp.getBlocks().size() == 1);

  return success();
}

void dynamatic::experimental::cfg::markBasicBlocks(
    handshake::FuncOp &funcOp, ConversionPatternRewriter &rewriter) {
  for (auto [blockID, block] : llvm::enumerate(funcOp)) {
    for (Operation &op : block) {
      if (!isa<handshake::MemoryOpInterface>(op)) {
        // Memory interfaces do not naturally belong to any block, so they do
        // not get an attribute
        op.setAttr(BB_ATTR_NAME, rewriter.getUI32IntegerAttr(blockID));
      }
    }
  }
}
