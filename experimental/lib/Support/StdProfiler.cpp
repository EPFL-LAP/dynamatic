//===- Simulator.cpp - std-level simulator ----------------------*- C++ -*-===//
//
// Implements a function profiler used (at this point) to generate block
// transition frequencies for smart buffer placement.
//
//===----------------------------------------------------------------------===//

#include "experimental/Support/StdProfiler.h"
#include "mlir/IR/Dominance.h"

#include <fstream>

using namespace mlir;
using namespace dynamatic::experimental;

/// Determines whether a string is a valid unsigned integer.
static bool isUnsigned(const std::string &str) {
  return std::all_of(str.begin(), str.end(),
                     [](char c) { return isdigit(c) || c == ' '; });
}

ArchBB::ArchBB(unsigned srcBB, unsigned dstBB, unsigned numTrans,
               bool isBackEdge)
    : srcBB(srcBB), dstBB(dstBB), numTrans(numTrans), isBackEdge(isBackEdge){};

StdProfiler::StdProfiler(mlir::func::FuncOp funcOp) : funcOp(funcOp){};

void StdProfiler::writeStats(bool printDOT) {
  mlir::raw_indented_ostream os(llvm::outs());
  if (printDOT)
    writeDOT(os);
  else
    writeCSV(os);
}

void StdProfiler::writeDOT(mlir::raw_indented_ostream &os) {
  // Print the graph
  os << "Digraph G {\n";
  os.indent();
  os << "splines=spline;\n";

  // Assign a unique name and create a DOT node for each block
  mlir::DenseMap<Block *, std::string> blockNames;
  for (auto [idx, block] : llvm::enumerate(funcOp.getBody())) {
    auto name = "block" + std::to_string(idx + 1);
    blockNames[&block] = name;
    os << "\"" << name << "\" [shape=box]\n";
  }

  for (auto &[blockPair, numTrans] : transitions)
    os << getDOTTransitionString(blockNames[blockPair.first],
                                 blockNames[blockPair.second], numTrans);

  // Assign a frequency of 0 to block transitions that never occured in the
  // input data
  for (auto &block : funcOp.getBody())
    for (auto *succ : block.getSuccessors())
      if (!transitions.contains(std::make_pair(&block, succ)))
        os << getDOTTransitionString(blockNames[&block], blockNames[succ], 0);

  os.unindent();
  os << "}\n";
}

void StdProfiler::writeCSV(mlir::raw_indented_ostream &os) {
  // Print column names
  os << "srcBlock,dstBlock,numTransitions,is_backedge\n";

  // Assign a unique id to each block based on their order of appearance in
  // the function
  mlir::DenseMap<Block *, unsigned> blockIDs;
  for (auto [idx, block] : llvm::enumerate(funcOp.getBody()))
    blockIDs[&block] = idx;

  // A block and its successor a backedge if the destination block dominates
  // the source block
  DominanceInfo domInfo(funcOp);

  for (auto &[blockPair, numTrans] : transitions)
    os << getCSVTransitionString(
        blockIDs[blockPair.first], blockIDs[blockPair.second], numTrans,
        domInfo.dominates(blockPair.second, blockPair.first));

  // Assign a frequency of 0 to block transitions that never occured in the
  // input data
  for (auto &block : funcOp.getBody())
    for (auto *succ : block.getSuccessors())
      if (!transitions.contains(std::make_pair(&block, succ)))
        os << getCSVTransitionString(blockIDs[&block], blockIDs[succ], 0,
                                     domInfo.dominates(succ, &block));
}

LogicalResult StdProfiler::readCSV(std::string &filename,
                                   SmallVector<ArchBB> &archs) {
  // Open a stream to the file
  std::ifstream inFile(filename);
  if (!inFile)
    return failure();

  std::string token;
  // Parses an unsigned integer from the given stream, stoping at the next comma
  // or at the stream's end. On success, sets the last argument to the parsed
  // value.
  auto parseToken = [&](std::istringstream &iss,
                        unsigned &value) -> ParseResult {
    std::getline(iss, token, ',');
    if (!isUnsigned(token))
      return failure();
    value = std::stoi(token) != 1;
    return success();
  };

  // Skip the header line
  std::string line;
  std::getline(inFile, line);

  // Parse lines one by one, creating an ArchBB for each
  while (std::getline(inFile, line)) {
    std::istringstream iss(line);

    // Parse all 4 columns
    unsigned srcBB, dstBB, numTrans, isBackEdge;
    if (parseToken(iss, srcBB) || parseToken(iss, dstBB) ||
        parseToken(iss, numTrans) || parseToken(iss, isBackEdge))
      return failure();

    // Add the arch to the list
    archs.emplace_back(srcBB, dstBB, numTrans, isBackEdge != 0);
  }
  return success();
}

std::string StdProfiler::getDOTTransitionString(std::string &srcBlock,
                                                std::string &dstBlock,
                                                unsigned freq) {
  return "\"" + srcBlock + "\" -> \"" + dstBlock +
         "\" [freq = " + std::to_string(freq) + "]\n";
}

std::string StdProfiler::getCSVTransitionString(unsigned srcBlock,
                                                unsigned dstBlock,
                                                unsigned freq,
                                                bool isBackedge) {
  return std::to_string(srcBlock) + "," + std::to_string(dstBlock) + "," +
         std::to_string(freq) + "," + (isBackedge ? "1" : "0") + "\n";
}
