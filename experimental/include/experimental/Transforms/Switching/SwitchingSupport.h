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
#include "mlir/IR/Attributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "experimental/Support/StdProfiler.h"
#include "dynamatic/Support/TimingModels.h"
#include "llvm/Support/Debug.h"

#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <vector>
#include <set>
#include <regex>
#include <string>
#include <cctype>
#include <typeinfo>
#include <optional>


using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

// Declaration
class AdjNode;
class AdjGraph;

// Helper datatype for switching estimation. Aggregates all useful information
// for the swithicng estimation process
struct SwitchingInfo {
  // This function insert (backedge pair, mgLabel) to the backEdgeToCFDFCMap
  void insertBE(unsigned srcBB, unsigned dstBB, StringRef mgLabel);

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
  std::map<unsigned, buffer::CFDFC> cfdfcs;
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

  // Map storing the subgraph of different segments in the dataflow circuit
  std::map<std::string, AdjGraph*> segToAdjGraphMap;

};


// Base class for the unit informaiton storing structure.
// This structure will be used throughout the estimation process
class NodeInfo {
  
};

// Class used to construct the per segment (MG & one-time execution segment)
// The class defined following is purely for path fining and preserve the structure
// of each subgraph.
/*
  Base Class used to store information related to node in the dataflow circuit,
        which will be used to calculate the switching activity in steady state.
            - Handshake channels
            - Data channels
        
        ! Handshake Signal Calculation:
            All child classes of this base class shall have it's own implementation of
                - cal_valid_switching
                - cal_ready_switching
                - cal_valid_set
                - cal_ready_set
        
            For the above functions, we have following assumptions:
                - Num switches must be >= 0.
                - For num switches, we use -1 to represent invalid value
                - For active set, we use None to represesnt invalid value
        
        ! Data Channel Calculation:
            For each node, we only store output data to its successors, like a directed chain list.
            Two possible situations for a given node:
                - It has only one (value, #iter) pair, like constant node. As the value never change
                - It has multiple (value, #iter) pair
            Thus, during dataout calculation, we need to check the len of the input data
            *If the node has multiple input data only two situations: 
                1) lengths they match with each other 
                2) one input has only 1 pair and the other has multiple.
            For the second case, the outputdata will have list length equal to the longer input data
*/
class AdjNode {
public:
  // Constructer
  AdjNode(mlir::Operation* selOp, 
          const std::vector<std::string>& predecessors, const std::vector<std::string>& successors, 
          const std::map<std::string, unsigned>& sucDataWidthMap, const unsigned& latency);

  // This funciton checks whether the handshake channel swiching calculation is finished or not
  bool handshakeFinished();

  // This function checks the handshake switching patterns of the node itself
  bool handshakeSwitchingChecking();

  // This function prints all information of the node
  void printDetail();

  // This function prints handshake switching
  void printHandshakeSwitching();

  // This function will calculate the total valid and ready swithcing for the node
  void totalHandshakeSwitchingUpdate();

  // This function updates the dataout process
  void updateDataout(int inputData);

  // 
  void updateHandshakeChannelSwitching(unsigned validChannelSwitching, unsigned readyChannelSwitching);

  // This function calculates the total number of switches of all data out channels
  //  Note: We treat "X" as invalid data and will count it only once
  void totalDataSwitchingCounting(bool mappedUnits);

  // This function will return the list of 1's position in the number's binary format
  std::vector<unsigned> getPositionList(int number);

  //
  void printDataChannelSwitching();
  void printPerDataChannelToggleNumber();
  void printPerHandshakeChannelToggleNumber();

  // Virtual class for switching calculation
  void calValidSwitching();
  void calReadySwitching();
  void calValdiSet();
  void calReadySet();

  // 
  //  Internal Storing Variables
  //
  unsigned nodeLatency = 0;       // Used to store the latency of the chosen node
  mlir::Operation* op;            // Pointer to the operation in the mlir file
  std::vector<std::string> pres;  // Vector storing the predecessors of the node in the segemnt
  std::vector<std::string> sucs;  // Vector storing the successors of the node in the segement

  // Handshake Signal Switching
  std::map<std::string, unsigned> validSignal;  // Map used to store number of switching of the node's valid signals (per channel, e.x., {"node_name" : 2})
  std::map<std::string, unsigned> readySignal;  // Map used to store number of switching of the node's ready signals (per channel, e.x., {"node_name" : 2})
  std::set<unsigned> setV;                      // Set used to store the active range of the corresponding valid signal in different channels
  std::set<unsigned> setR;                      // Set used to store active range of the corresponding ready signal in different channels

  // Data channel Switching
  std::map<std::string, unsigned> sucsDataWidthMap; // Map from succeeding node name to the channel width
  std::map<std::string, std::map<unsigned, unsigned>> perChannelToggle; // toggle count dict per bitwidth
  std::map<std::string, std::vector<int>> dataOut;  // Map from succeeding node name to the corresponding outptu channel dict
  
  // Overall signal switching status
  std::map<std::string, std::vector<unsigned>> dataSwitches;  // Map storing the number of data switches for different data channels
  std::map<std::string, unsigned> handshakeSwitches;          // Map storing the number of switches in different handshake channels

  unsigned totalValidSwitching = 0;
  unsigned totalReadySwitching = 0;
  unsigned totalDataSwitching = 0;

  // Update Flag
  bool handshakeUpdateFlag = false;
};

// Seg subgraph that stores all the nodes of a segment in the dataflow circuit
class AdjGraph {
public:
  AdjGraph(const buffer::CFDFC& cfdfc, const TimingDatabase& timingDB, unsigned II);

  // 
  //  Internal Storing Variables
  //
  std::vector<std::string> segStartNodes;             // Vector storing all starting nodes in the segment
  std::map<std::string, AdjNode> nodes;               // Map from unit name to the corresponding node storing structure

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

// Internal timing database for all handshake ops
const std::map<std::string, unsigned> OP_DELAY_MAP = {
  {"handshake.source", 0},
  {"handshake.cmpi", 0},
  {"handshake.addi", 0},
  {"handshake.subi", 0},
  {"handshake.muli", 4},
  {"handshake.extsi", 0},
  {"handshake.mc_load", 1},
  {"handshake.mc_store", 0},
  {"handshake.lsq_load", 5},
  {"handshake.lsq_store", 0},
  {"handshake.merge", 0},
  {"handshake.addf", 10},
  {"handshake.subf", 10},
  {"handshake.mulf", 6},
  {"handshake.divui", 36},
  {"handshake.divsi", 36},
  {"handshake.divf", 30},
  {"handshake.cmpf", 2},
  {"handshake.control_merge", 0},
  {"handshake.fork", 0},
  {"handshake.d_return", 0},
  {"handshake.cond_br", 0},
  {"handshake.end", 0},
  {"handshake.andi", 0},
  {"handshake.ori", 0},
  {"handshake.xori", 0},
  {"handshake.shli", 0},
  {"handshake.shrsi", 0},
  {"handshake.shrui", 0},
  {"handshake.select", 0},
  {"handshake.mux", 0},
  {"handshake.source", 0},
  {"handshake.trunci", 0},
  {"handshake.constant", 0},
  {"handshake.extui", 0},
  {"handshake.mem_controller", 0}
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
