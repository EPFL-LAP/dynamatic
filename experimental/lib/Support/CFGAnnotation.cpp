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
#include "llvm/Support/Debug.h"
#include <regex>

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

  /// The CFG annotation must be present in the input function
  if (!funcOp->hasAttr(cfg::CFG_EDGES)) {
    return funcOp->emitError() << "Handshake function has no annotation `"
                               << cfg::CFG_EDGES << "`\n";
  }
  auto edges = funcOp->getAttrOfType<mlir::StringAttr>(cfg::CFG_EDGES).str();

  /// Function to get all the info of the edges in the function
  auto getBracketsContent =
      [](const std::string &input) -> std::vector<std::string> {
    std::vector<std::string> results;
    std::regex pattern(R"(\[(.*?)\])");
    std::smatch matches;

    std::string::const_iterator searchStart(input.cbegin());
    while (std::regex_search(searchStart, input.cend(), matches, pattern)) {
      results.push_back(matches[1]);
      searchStart = matches.suffix().first;
    }

    return results;
  };

  using condBranch_t =
      std::optional<std::tuple<unsigned, unsigned, unsigned, std::string>>;
  using uncondBranch_t = std::optional<std::tuple<unsigned, unsigned>>;

  /// Function to parse an unconditional branch
  auto parseUnconditional = [&](const std::string &input) -> uncondBranch_t {
    std::regex pattern(R"((\d+),(\d+))");
    std::smatch matches;

    if (std::regex_match(input, matches, pattern))
      return uncondBranch_t({std::stoul(matches[1]), std::stoul(matches[2])});

    return {};
  };

  /// Function to parse a conditional branch
  auto parseConditional = [&](const std::string &input) -> condBranch_t {
    std::regex pattern(R"((\d+),(\d+),(\d+),([a-zA-Z0-9]+))");
    std::smatch matches;

    if (std::regex_match(input, matches, pattern)) {
      return condBranch_t({std::stoul(matches[1]), std::stoul(matches[2]),
                           std::stoul(matches[3]), matches[4]});
    }

    return {};
  };

  cfg::CFGAnnotation result;

  // For each edge, try to parse as a conditional and unconditional branch
  for (auto &edge : getBracketsContent(edges)) {
    if (auto uncond = parseUnconditional(edge); uncond.has_value()) {
      auto br = uncond.value();
      result.insert({std::get<0>(br), cfg::CFGNode(std::get<1>(br))});
    } else if (auto cond = parseConditional(edge); cond.has_value()) {
      auto br = cond.value();
      result.insert(
          {std::get<0>(br),
           cfg::CFGNode(std::get<1>(br), std::get<2>(br), std::get<3>(br))});
    } else {
      return funcOp->emitError()
             << "Format of CFG annotation is not correct the following edge: "
             << edge << "\n";
    }
  }

  return result;
}

/// Extract the CFG information from a region. Each basic block with an out
/// edge is marked with the successor basic blocks, related to the structure
/// `CFGEdge` for edges.
static cfg::CFGAnnotation getCFGEdges(Region &funcRegion, NameAnalysis &namer) {

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
      std::string conditionName = namer.getName(conditionOperation).str();

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

void dynamatic::experimental::cfg::annotateCFG(handshake::FuncOp &funcOp,
                                               PatternRewriter &rewriter,
                                               NameAnalysis &namer) {

  // Get the CFG information
  const auto edgeMap = getCFGEdges(funcOp.getRegion(), namer);

  std::string result = "";

  auto appendSymbol = [&](char s) { result += s; };
  auto appendStr = [&](const std::string &s) { result += s; };

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

  funcOp->setAttr(CFG_EDGES, rewriter.getStringAttr(result));
}

LogicalResult
dynamatic::experimental::cfg::restoreCfStructure(handshake::FuncOp &funcOp,
                                                 PatternRewriter &rewriter) {

  // Get an operation according to its name. If not present, return a compare
  // operation in the block provided as input
  auto getOpByName = [&](const std::string &name,
                         unsigned blockID) -> Operation * {
    constexpr llvm::StringLiteral nameAttr("handshake.name");
    Operation *cmpSameBlock = nullptr;
    std::string otherName;

    // For each operaiton
    for (Operation &op : funcOp.getOps()) {

      // Continue if the operation has no name attribute
      if (!op.hasAttr(nameAttr))
        continue;

      // Return the operation if it has the same name as the input
      auto opName = op.getAttrOfType<mlir::StringAttr>(nameAttr);
      std::string opNameStr = opName.str();

      // Save the operation if it is a compare (both integer and floating) in
      // the same block of the provdied one
      if (llvm::isa_and_nonnull<handshake::CmpIOp, handshake::CmpFOp>(op) &&
          getLogicBB(&op).value() == blockID) {
        otherName = opNameStr;
        cmpSameBlock = &op;
      }

      if (name == opNameStr)
        return &op;
    }

    llvm::errs() << "[WARNING][CFG] Operation `" << name
                 << "` cannot be found; using `" << otherName
                 << "` instead. Correctness in not guaranteed\n";

    return cmpSameBlock;
  };

  // Get the map of edges
  auto failureOrEdges = unserializeEdges(funcOp);
  if (failed(failureOrEdges))
    return failure();
  CFGAnnotation &edges = *failureOrEdges;

  // List of the original blocks in the CFG, obtained from the list of edges.
  // The final node (without any outgoing edge) is added as well.
  SmallVector<unsigned> blocksList;
  for (auto &[source, _] : edges)
    blocksList.push_back(source);
  std::sort(blocksList.begin(), blocksList.end());
  blocksList.push_back(blocksList.back() + 1);

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

  for (auto blockIndex : blocksList) {

    // For each operation
    for (auto *op : originalOps) {

      if (!cannotBelongToCFG(op) && !getLogicBB(op))
        op->emitError() << "Operation should have basic block attribute.";

      // Get block ID of the current operation. If it is not annotated (end
      // operation/LSQ/memory operation) then use the block 0.
      unsigned opBlock = getLogicBB(op).value_or(0);

      // Move the current operation at the end of the block
      if (opBlock == blockIndex)
        op->moveBefore(currentBlock, currentBlock->end());
    }

    // Once all the operations have been covered for a given bb, we move to the
    // following.
    if (blockIndex != blocksList.back()) {
      currentBlock = funcOp.addBlock();
      indexToBlock.insert({blockIndex + 1, currentBlock});
    }
  }

  // Once we are done creating the blocks, we need to insert the terminators
  // to obtain a proper block structure, using the edge information provided
  // as input

  // For each block
  for (auto [blockID, bb] : llvm::enumerate(funcOp.getBlocks())) {
    rewriter.setInsertionPointToEnd(&bb);

    if (!edges.contains(blockID)) {
      rewriter.create<func::ReturnOp>(bb.back().getLoc());
      continue;
    }

    auto edge = edges.lookup(blockID);

    // Either create a conditional or unconditional branch depending on the
    // type of edge we have
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

void dynamatic::experimental::cfg::markBasicBlocks(handshake::FuncOp &funcOp,
                                                   PatternRewriter &rewriter) {
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
