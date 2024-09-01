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
#include <set>
#include <regex>
#include <string>
#include <cctype>


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
  std::unordered_map<unsigned, float_t> cfdfcIIs;
  // Map from CFDFC index to the CFDFC info stroing class
  // Only contain the number of edges, backedges etc.
  // No info about unit delay, node neighbors etc.
  std::map<unsigned, buffer::CFDFC*> cfdfcs;
  // Set of names of the ALUs in the Handshake level mlir file
  // Used to retrieve the date from SCF level profiling
  std::vector<std::string> funcOpNames;

  // Map from Backedge pair to the list of CFDFC lable vector
  // i.e. {(1, 1) : [1]}
  std::map<std::pair<unsigned, unsigned>, std::vector<unsigned>> backEdgeToCFDFCMap;
  // Map from Segment Label (CFDFC and temporal transaction sections that are not MGs)
  std::map<std::string, std::vector<unsigned>> segToBBListMap;
  // Map from Transaction Segment label to the successing MG label
  std::map<std::string, std::string> transToSucMGMap;

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
// Unique Sets for the parsing process
//
//===----------------------------------------------------------------------===//
// Define the constant name sensitive list used for parsing the profiling results
// As the name of the same operation in scf level IR and the final handshake IR
// is different, we need to map the scf level op to the handshake mlir file.
// TODO: Solve this more elegantly, directly add the scf level name in the attribute
// TODO: Add the name of the rest of operations, i.e. lsq etc.
const std::set<std::string> NAME_SENSE_LIST = {
  "muli",
  "addi",
  "subi",
  "ori",
  "andi",
  "cmpi",
  "mc_load",
  "mc_store",
  "lsq_load",
  "lsq_store",
  "load",
  "store",
  "shli",
  "shrsi"
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

// This function prints all values in a vector
template <typename T>
inline void printVector(const T& selVec) {
  int counter = 0;

  llvm::dbgs() << "[DEBUG] Vector Contents: "; 

  for (auto& selVal : selVec) {
    llvm::dbgs() << "[" << counter << "] : " << selVal << " ";

    counter++;
  }

  llvm::dbgs() << ";\n";
}

// This function remove the digits in the given string and keep the rest
std::string removeDigits(const std::string& inStr);

// This function split a given string into a vector based on the delimiter
std::vector<std::string> split(const std::string &s, const std::string& delimiter);

// This function removes the starting and ending empty space
std::string strip(const std::string &inputStr, const std::string &toRemove);


#endif // EXPERIMENTAL_TRANSFORMS_SWITCHING_SUPPORT_H
