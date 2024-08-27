//===- SwitchingSupport.cpp - Estimate Swithicng Activities ------*- C++ -*-===//
//
// Implements the supporting datastructures for the switching estimation
// pass
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/Switching/SwitchingSupport.h"

using namespace mlir;
using namespace dynamatic;

void SwitchingInfo::insertBE(unsigned srcBB, unsigned dstBB, StringRef mgLabel) {
  std::pair<unsigned, unsigned> BBPair = {srcBB, dstBB};

  // Check the existence of the backedge pair
  if (backEdgeToCFDFCMap.find(BBPair) != backEdgeToCFDFCMap.end()) {
    backEdgeToCFDFCMap[BBPair].push_back(static_cast<unsigned>(std::stoul(mgLabel.str())));
  } else {
    std::vector<unsigned> tmpVector{static_cast<unsigned>(std::stoul(mgLabel.str()))};
    backEdgeToCFDFCMap[std::make_pair(srcBB, dstBB)] = tmpVector;
  }
}

//===----------------------------------------------------------------------===//
//
// Helper Functions
//
//===----------------------------------------------------------------------===//

void printBEToCFDFCMap(const std::map<std::pair<unsigned, unsigned>, std::vector<unsigned>>& selMap) {
  for (const auto& selPair : selMap) {
    const std::pair<unsigned, unsigned>& key = selPair.first;
    const std::vector<unsigned> mgList = selPair.second;

    llvm::dbgs() << "[DEBUG] \tBackEdge Pair: (" << key.first << ", " << key.second <<") : [";

    for (const auto& selMG: mgList) {
      llvm::dbgs() << selMG << ", ";
    }

    llvm::dbgs() << "]\n";
  }
}

void printSegToBBListMap(const std::map<std::string, mlir::SetVector<unsigned>>& selMap) {
  for (const auto& selPair: selMap) {
    const std::string segLabel = selPair.first;
    const mlir::SetVector<unsigned> BBList = selPair.second;

    llvm::dbgs() << "[DEBUG] \tSeg Label: " << segLabel <<" : [";

    for (const auto& selBB: BBList) {
      llvm::dbgs() << selBB << ", ";
    }

    llvm::dbgs() << "]\n";
  }
}


