//===- SwitchingSupport.h - Switching Estimation -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares supporting data structures for the swithcing estimation pass
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_TRANSFORMS_SWITCHING_SUPPORT_H
#define EXPERIMENTAL_TRANSFORMS_SWITCHING_SUPPORT_H

#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "experimental/Support/StdProfiler.h"
#include "llvm/Support/Debug.h"

#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <vector>


using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

// Helper datatype for switching estimation. Aggregates all useful information
// for the swithicng estimation process
struct SwitchingInfo {
  // This function insert (backedge pair, mgLabel) to the backEdgeToCFDFCMap
  void insertBE(unsigned srcBB, unsigned dstBB, StringRef mgLabel);

  // This function insert (segLabel, BBlist) to segToBBListMap
  void insertSeg();


  // 
  //  Internal Storing Variables
  //
  // All backedge BB pairs in the dataflow circuit
  llvm::SmallVector<std::pair<unsigned, unsigned>> backEdges;
  // Map from CFDFC index to the corresponding II values
  std::unordered_map<int, float_t> cfdfcIIs;
  // Map from CFDFC index to the CFDFC info stroing class
  // Only contain the number of edges, backedges etc.
  // No info about unit delay, node neighbors etc.
  std::map<int, buffer::CFDFC> cfdfcs;

  // Map from Backedge pair to the list of CFDFC lable vector
  // i.e. {(1, 1) : [1]}
  std::map<std::pair<unsigned, unsigned>, std::vector<unsigned>> backEdgeToCFDFCMap;
  // Map from Segment Label (CFDFC and temporal transaction sections that are not MGs)
  std::map<std::string, mlir::SetVector<unsigned>> segToBBListMap;

};


// Base class for the unit informaiton storing structure.
// This structure will be used throughout the estimation process
class NodeInfo {
  
};

// Class used to construct the per segment (MG & one-time execution segment)
// The class defined following is purely for path fining and preserve the structure
// of each subgraph.
// Other information about the signal switching etc. should be stored in NodeInfo instead.
class AdjNode {

};



//===----------------------------------------------------------------------===//
//
// Helper Functions
//
//===----------------------------------------------------------------------===//
// The following function prints the backEdgeToCFDFCMap 
void printBEToCFDFCMap(const std::map<std::pair<unsigned, unsigned>, std::vector<unsigned>>& selMap);

// This function prints the Segment ID to BBlist map
void printSegToBBListMap(const std::map<std::string, mlir::SetVector<unsigned>>& selMap);

#endif // EXPERIMENTAL_TRANSFORMS_SWITCHING_SUPPORT_H
