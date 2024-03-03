//===- SharingSupport.h - Resource Sharing Utilities-----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains,
// 1) the data structures needed to implement the resource sharing algorthm
// 2) functions to generate performance models
// 3) overload the HandshakePlaceBuffersPass to get performance information
//    and to inject constraints that model ordered accesses of shared resources
//
//===----------------------------------------------------------------------===//
#ifndef EXPERIMENTAL_INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_SHARINGSUPPORT_H
#define EXPERIMENTAL_INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_SHARINGSUPPORT_H

#include <utility>

#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/FPGA20Buffers.h"
#include "experimental/Transforms/ResourceSharing/SCC.h"

using namespace dynamatic;
using namespace dynamatic::handshake;

using namespace dynamatic::experimental::sharing;

namespace dynamatic {
namespace experimental {
namespace sharing {

/*
 *  stores/transfers information needed for resource sharing from buffer
 * placement
 */
struct ResourceSharingInfo {
  // for each CFDFC, store the throughput in double format to double format to
  // compare
  std::map<int, double> throughputPerCFDFC{};

  // stores shareable operations and their occupancy
  struct OperationData {
    mlir::Operation *op;
    double occupancy;

    void print();
  };
  std::vector<OperationData> operations;

  // used to perform SCC-computation (finding strongly connected components)
  SmallVector<dynamatic::experimental::ArchBB> archs;

  handshake::FuncOp funcOp;

  // list of values where to insert a seq buffer
  std::vector<Value> opaqueChannel = {};

  // determines if one should give the full report back or just the current
  // occupancy sum
  bool fullReportRequired = true;

  // specific cluster of operations
  std::set<Operation *> testedGroups;
  // occupancy sum of specific cluster of operations (see above)
  double occupancySum;
  void computeOccupancySum();

  // constructor
  ResourceSharingInfo() = default;
};

/*
       Inside each set of strongly connected components
       one needs to check if sharing is possible
*/
struct Group {
  std::vector<mlir::Operation *> items;
  double sharedOccupancy;
  bool hasCycle;

  // determine if an operation is cyclic (if there is a path from the op that
  // reaches back to it)
  bool recursivelyDetermineIfCyclic(mlir::Operation *currentOp,
                                    std::set<mlir::Operation *> &nodeVisited,
                                    mlir::Operation *op);
  bool determineIfCyclic(mlir::Operation *op);

  // add operation to group
  // important: changes neither the shared occupancy nor the hasCycle attributes
  void addOperation(mlir::Operation *op);

  bool operator<(const Group& a) const {
    return sharedOccupancy < a.sharedOccupancy;
  }

  bool operator==(Group a) const {
    if (items.empty()) {
      return true;
    }
    return items[0] == a.items[0];
  }

  Group &operator=(const Group &a) {
    items = a.items;
    sharedOccupancy = a.sharedOccupancy;
    hasCycle = a.hasCycle;
    return *this;
  }

  // Constructors
  Group(std::vector<mlir::Operation *> ops, double occupancy, bool cyclic)
      : sharedOccupancy(occupancy) {
    items = std::move(ops);
    hasCycle = cyclic;
  }
  Group(mlir::Operation *op, double occupancy) : sharedOccupancy(occupancy) {
    items.push_back(op);
    hasCycle = determineIfCyclic(op);
  }
  Group(mlir::Operation *op) : sharedOccupancy(-1) {
    items.push_back(op);
    hasCycle = determineIfCyclic(op);
  }

  // Destructor
  ~Group();
};

// abbreviation to iterate through list of groups
using GroupIt = std::list<Group>::iterator;

/*
       Each basic block resides in a set
       of strongly connected components
       For Example: A set could be {1,2,3}
*/
struct Set {
  std::list<Group> groups{};
  int sccId;
  double opLatency;

  // add group to set
  void addGroup(const Group& group);

  // merge two existing groups (of this specific set) into one newly created
  // group
  void joinGroups(GroupIt group1, GroupIt group2,
                  std::vector<mlir::Operation *> &finalOrd);

  // join another set to this specific set
  // important: while joining, one group of each set is paired with one group of
  // the other set
  void joinSet(Set *joinedElement);

  // print content of specific set
  void print();

  // Constructors
  Set(double latency) { opLatency = latency; }

  Set(const Group& group) { groups.push_back(group); }

  Set(int sccIdx, double latency) {
    sccId = sccIdx;
    opLatency = latency;
  }
};

/*
       Each operation type (e.g. mul, add, load)
       can be treated separately
*/
struct ResourceSharingForSingleType {
  double opLatency;
  llvm::StringRef identifier;
  std::vector<Set> sets{};
  std::map<int, int> setSelect;
  Set finalGrouping;
  std::list<mlir::Operation *> opsNotOnCfg;

  // add set to operation type
  void addSet(const Group& group);

  // print the composition of Sets/SCCs - Groups
  void print();

  // the end of the sharing strategy joins sets together, use this to print the
  // final set
  void printFinalGroup();

  // joines the sets at the end of the sharing algorithm
  void sharingAcrossLoopNests();

  // joines operations that do not reside in a CFDFC
  void sharingOtherUnits();

  // Constructor
  ResourceSharingForSingleType(double latency, llvm::StringRef identifier)
      : opLatency(latency), identifier(identifier),
        finalGrouping(Set(latency)) {}
};

struct ControlStructure {
  mlir::Value controlMerge;
  mlir::Value controlBranch;
  mlir::Value currentPosition;
};

/*
       Class to iterate easily trough all
       operation types
*/
class ResourceSharing {
  // troughput per CFDFC
  std::map<int, double> throughput;
  // connections between basic blocks
  SmallVector<dynamatic::experimental::ArchBB> archs;
  // maps operation types to integers (SCC analysis)
  std::map<llvm::StringRef, int> opNames;
  // number of sharable operation types
  int numberOfOperationTypes;
  // operation directly after start
  Operation *firstOp = nullptr;
  // Operations in topological order
  std::map<Operation *, unsigned int> opTopologicalOrder;

  // used to run topological sorting
  void recursiveDFStravel(Operation *op, unsigned int *position,
                          std::set<mlir::Operation *> &nodeVisited);

public:
  // stores control merge and branch of each BB
  std::map<int, ControlStructure> controlMap;

  std::vector<ResourceSharingForSingleType> operationTypes;

  // set first operation of the IR
  void setFirstOp(Operation *op);

  // get first operation of the IR
  Operation *getFirstOp();

  // compute first operation of the IR
  bool computeFirstOp(handshake::FuncOp funcOp);

  // calculate the topological ordering of all operations
  // important: operations on a cycle do not have a topological order
  //            but are still present
  void initializeTopolocialOpSort(handshake::FuncOp *funcOp);

  // print operations in topological order
  void printTopologicalOrder();

  // sort operations in two groups topologically
  std::vector<Operation *> sortTopologically(GroupIt group1, GroupIt group2);

  // determine if a vector of operations are in topological order
  bool isTopologicallySorted(std::vector<Operation *> ops);

  // place resource sharing data retrieved from buffer placement
  void retrieveDataFromPerformanceAnalysis(const ResourceSharingInfo& sharingFeedback,
                                           std::vector<int> &scc,
                                           int numberOfScc,
                                           const TimingDatabase& timingDB);

  // return number of Basic Blocks
  int getNumberOfBasicBlocks();

  // place retrieved connections between Basic blocks
  void getListOfControlFlowEdges(
      SmallVector<dynamatic::experimental::ArchBB> archsExt);

  // perform SCC-agorithm on basic block level
  std::vector<int> performSccBbl();

  // perform SCC-agorithm on operation level
  void performSccOpl(std::set<mlir::Operation *> &result,
                      handshake::FuncOp *funcOp);

  // print source-destination BB of connection between BBs, throughput per CFDFC
  // and the composition in operation-type, set, group
  void print();

  // find control structure of each BB: control_merge, control_branch
  void getControlStructure(handshake::FuncOp funcOp);

  // place and compute all necessary data to perform resource sharing
  void
  placeAndComputeNecessaryDataFromPerformanceAnalysis(ResourceSharingInfo data,
                                                      const TimingDatabase& timingDB);

  // constructor
  ResourceSharing(ResourceSharingInfo data, const TimingDatabase& timingDB) {
    placeAndComputeNecessaryDataFromPerformanceAnalysis(std::move(data), timingDB);
  }
};

// create all possible pairs of Groups in a specific set
// Use a struct to define the comparison operator
std::vector<std::pair<GroupIt, GroupIt>>
combinations(Set *set, std::map<Group, std::set<Group>> &alreadyTested);

// test if two doubles are equal
bool equal(double a, double b);

// test if a double is less or equal than an other double
bool lessOrEqual(double a, double b);

// generate performance model with all neccessary connections
Value generatePerformanceStep(OpBuilder *builder, mlir::Operation *op,
                                std::map<int, ControlStructure> &controlMap);
std::vector<Value>
generatePerformanceModel(OpBuilder *builder,
                           std::vector<mlir::Operation *> &items,
                           std::map<int, ControlStructure> &controlMap);

// revert IR to its original state
void revertPerformanceStep(OpBuilder *builder, mlir::Operation *op);
void destroyPerformanceModel(OpBuilder *builder,
                               std::vector<mlir::Operation *> &items);

// extend the current fork by one output and return that output
mlir::OpResult extendFork(OpBuilder *builder, handshake::ForkOp oldFork);

// add sink at the given connection point
void addSink(OpBuilder *builder, mlir::OpResult *connectionPoint);

// add constant at the given connection point
mlir::OpResult addConst(OpBuilder *builder, mlir::OpResult *connectionPoint,
                        int value);

// add branch at the given connection point
mlir::OpResult addBranch(OpBuilder *builder, mlir::OpResult *connectionPoint);

// delete all buffers in the IR
void deleteAllBuffers(handshake::FuncOp funcOp);

} // namespace sharing
} // namespace experimental
} // namespace dynamatic

namespace permutation {

using PermutationEdge = std::vector<Operation *>::iterator;

// input: vector of Operations
// changes input: sort vector in BB regions and sort those with regular
// definition of "less" in Operation class output: starting and ending
// operations of every present basic block
/*
 * example: state: BB1{Op1, Op2}, BB2{Op3, Op4, Op5}
 *          input: {Op3, Op2, Op1, Op5, Op4}
 *          change to: {Op1, Op2, Op3, Op4, Op5}
 *          output: {0,2},{2,5}
 */
void findBBEdges(std::deque<std::pair<int, int>> &bbOps,
                 std::vector<Operation *> &permutationVector);

// inputs: permutation_vector.begin(), output of function findBBEdges
// changes: permutation_vector to the next permutation step
// output: false if all permuations visited, else true
/*
 * further information: This is a extended version of next_permutation in
 * package algorithm. As we do not need to permute operations of different basic
 * blocks, this function does exactly only permute within a BB region and goes
 * over all combinations of permutations of different BBs.
 */
bool getNextPermutation(PermutationEdge beginOfPermutationVector,
                          std::deque<std::pair<int, int>> &separationOfBBs);

} // namespace permutation

namespace dynamatic {
namespace buffer {
namespace fpga20 {

class MyFPGA20Buffers : public FPGA20Buffers {
public:
  std::vector<ResourceSharingInfo::OperationData> getData();
  double getOccupancySum(std::set<Operation *> &group);
  LogicalResult addSyncConstraints(const std::vector<Value>& opaqueChannel);

  // constructor
  MyFPGA20Buffers(GRBEnv &env, FuncInfo &funcInfo,
                  const TimingDatabase &timingDB, double targetPeriod,
                  bool legacyPlacement, Logger &logger, StringRef milpName)
      : FPGA20Buffers(env, funcInfo, timingDB, targetPeriod, legacyPlacement,
                      logger, milpName){

        };
};

} // namespace fpga20
} // namespace buffer
} // namespace dynamatic

#endif // EXPERIMENTAL_INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_SHARINGSUPPORT_H