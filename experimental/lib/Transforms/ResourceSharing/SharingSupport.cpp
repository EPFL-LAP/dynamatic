//===- SharingSupport.cpp - Resource Sharing Utilities-----------*- C++ -*-===//
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
#include <utility>

#include "experimental/Transforms/ResourceSharing/SharingSupport.h"

using namespace dynamatic::handshake;
using namespace dynamatic::buffer::fpga20;
using namespace dynamatic::experimental::sharing;

void ResourceSharingInfo::OperationData::print() {
  llvm::errs() << "Operation " << op << ", occupancy: " << occupancy
               << ", block: " << getLogicBB(op) << "\n";
}


void Group::addOperation(mlir::Operation *op) { items.push_back(op); }

bool Group::recursivelyDetermineIfCyclic(
    mlir::Operation *currentOp, std::set<mlir::Operation *> &nodeVisited,
    mlir::Operation *op) {
  nodeVisited.insert(currentOp);
  for (auto &u : currentOp->getResults().getUses()) {
    Operation *childOp = u.getOwner();
    if (childOp == op) {
      return true;
    }
    auto it = nodeVisited.find(childOp);
    if (it == nodeVisited.end()) {
      // not visited yet
      if (recursivelyDetermineIfCyclic(childOp, nodeVisited, op)) {
        return true;
      }
    }
  }
  return false;
}

bool Group::determineIfCyclic(mlir::Operation *op) {
  std::set<mlir::Operation *> nodeVisited;
  return recursivelyDetermineIfCyclic(op, nodeVisited, op);
}

void Set::addGroup(const Group& group) { groups.push_back(group); }

void Set::joinGroups(GroupIt group1, GroupIt group2,
                     std::vector<mlir::Operation *> &finalOrd) {
  Group newlyCreated =
      Group(finalOrd, group1->sharedOccupancy + group1->sharedOccupancy,
            group1->hasCycle || group2->hasCycle);
  groups.erase(group1);
  groups.erase(group2);
  groups.push_back(newlyCreated);
}

void Set::joinSet(Set *joinedElement) {
  GroupIt pelem = groups.begin();
  for (GroupIt jelem = joinedElement->groups.begin();
       jelem != joinedElement->groups.end(); pelem++, jelem++) {
    pelem->items.insert(pelem->items.end(), jelem->items.begin(),
                        jelem->items.end());
  }
}
void Set::print() {
  llvm::errs() << "Set id: " << sccId << "\n";
  int i = 0;
  for (const auto& group : groups) {
    llvm::errs() << "Group #" << i++ << "\n";
    for (auto *op : group.items) {
      llvm::errs() << op << ", ";
    }
    llvm::errs() << "\n";
  }
}

void ResourceSharingForSingleType::addSet(const Group& group) {
  sets.emplace_back(group);
}

void ResourceSharingForSingleType::print() {
  llvm::errs() << identifier << "\n";
  for (const auto& set : sets) {
    llvm::errs() << "SCC" << set.sccId << ":\n";
    int groupCount = 0;
    for (const auto& group : set.groups) {
      llvm::errs() << "Group " << groupCount++ << ": ";
      for (auto *item : group.items) {
        llvm::errs() << item << ", ";
      }
    }
    llvm::errs() << "\n";
  }
}

void ResourceSharingForSingleType::printFinalGroup() {
  llvm::errs() << "Final grouping for " << identifier << ":\n";
  int groupCount = 0;
  for (const auto& group : finalGrouping.groups) {
    llvm::errs() << "Group " << groupCount++ << ": ";
    for (auto *item : group.items) {
      llvm::errs() << item << ", ";
    }
  }
  llvm::errs() << "\n";
}

void ResourceSharingForSingleType::sharingAcrossLoopNests() {
  int numberOfSets = sets.size();
  if (!numberOfSets) {
    return;
  }

  int maxSetSize = -1;
  int maxIdx = -1;
  for (int i = 0; i < numberOfSets; i++) {
    if ((int)sets[i].groups.size() > maxSetSize) {
      maxSetSize = sets[i].groups.size();
      maxIdx = i;
    }
  }
  // choose initial set
  finalGrouping = sets[maxIdx];

  for (int i = 0; i < numberOfSets; i++) {
    if (i == maxIdx) {
      continue;
    }
    finalGrouping.joinSet(&sets[i]);
  }
}

void ResourceSharingForSingleType::sharingOtherUnits() {
  auto it = finalGrouping.groups.begin();
  for (auto *unit : opsNotOnCfg) {
    it->addOperation(unit);
    it++;
    if (it == finalGrouping.groups.end()) {
      it = finalGrouping.groups.begin();
    }
  }
}

void ResourceSharing::recursiveDFStravel(
    Operation *op, unsigned int *position,
    std::set<mlir::Operation *> &nodeVisited) {
  // add operation
  nodeVisited.insert(op);

  // DFS over all child ops
  for (auto &u : op->getResults().getUses()) {
    Operation *childOp = u.getOwner();
    auto it = nodeVisited.find(childOp);
    if (it == nodeVisited.end()) {
      // not visited yet
      recursiveDFStravel(childOp, position, nodeVisited);
    }
  }
  // update container
  opTopologicalOrder[op] = *position;
  ++(*position);
}

bool ResourceSharing::computeFirstOp(handshake::FuncOp funcOp) {
  // If we are in the entry block, we can use the start input of the
  // function (last argument) as our control value
  if (!funcOp.getArguments().back().getType().isa<NoneType>()) {
    return false;
  }
  Value func = funcOp.getArguments().back();
  std::vector<Operation *> startingOps;
  for (auto &u : func.getUses())
    startingOps.push_back(u.getOwner());
  if (startingOps.size() != 1) {
    return false;
  }
  firstOp = startingOps[0];
  return true;
}

Operation *ResourceSharing::getFirstOp() { return firstOp; }

void ResourceSharing::initializeTopolocialOpSort(handshake::FuncOp *funcOp) {
  if (firstOp == nullptr) {
    llvm::errs() << "[Error] Operation directly after start not yet present\n";
  }
  unsigned int position = 0;
  std::set<mlir::Operation *> nodeVisited;
  recursiveDFStravel(firstOp, &position, nodeVisited);
  for (Operation &op : funcOp->getOps()) {
    auto it = nodeVisited.find(&op);
    if (it == nodeVisited.end()) {
      recursiveDFStravel(&op, &position, nodeVisited);
    }
  }
}

void ResourceSharing::printTopologicalOrder() {
  llvm::errs() << "Topological Order: \n";
  for (auto [op, id] : opTopologicalOrder) {
    llvm::errs() << id << " : " << op << "\n";
  }
}

std::vector<Operation *> ResourceSharing::sortTopologically(GroupIt group1,
                                                            GroupIt group2) {
  std::vector<Operation *> result(group1->items.size() + group2->items.size());
  // add all operations in sorted order
  merge(group1->items.begin(), group1->items.end(), group2->items.begin(),
        group2->items.end(), result.begin(),
        [this](Operation *a, Operation *b) {
          return opTopologicalOrder[a] > opTopologicalOrder[b];
        });
  return result;
}

bool ResourceSharing::isTopologicallySorted(std::vector<Operation *> ops) {
  for (unsigned long i = 0; i < ops.size() - 1; i++) {
    if (opTopologicalOrder[ops[i]] > opTopologicalOrder[ops[i + 1]]) {
      return false;
    }
  }
  return true;
}

void ResourceSharing::retrieveDataFromPerformanceAnalysis(
    const ResourceSharingInfo& sharingFeedback, std::vector<int> &scc,
    int numberOfScc, const TimingDatabase& timingDB) {
  // take biggest occupancy per operation
  std::unordered_map<mlir::Operation *, double> uniqueOperation;
  for (auto item : sharingFeedback.operations) {
    if (uniqueOperation.find(item.op) != uniqueOperation.end()) {
      // operation already present
      uniqueOperation[item.op] =
          std::max(item.occupancy, uniqueOperation[item.op]);
    } else {
      // add operation
      uniqueOperation[item.op] = item.occupancy;
    }
  }

  // everytime we place/overwrite data, initial number of operation types is 0;
  numberOfOperationTypes = 0;

  // iterate through all retrieved operations
  for (auto op : uniqueOperation) {
    // choose the right operation type
    double latency;
    if (failed(timingDB.getLatency(op.first, SignalType::DATA, latency)))
      latency = 0.0;

    llvm::StringRef opName = op.first->getName().getStringRef();
    Group groupItem = Group(op.first, op.second);
    int opIdx = -1;
    auto item = opNames.find(opName);
    if (item != opNames.end()) {
      opIdx = item->second;
    } else {
      opNames[opName] = numberOfOperationTypes;
      opIdx = numberOfOperationTypes;
      ++numberOfOperationTypes;
      operationTypes.emplace_back(latency, opName);
    }
    ResourceSharingForSingleType &opT = operationTypes[opIdx];

    // choose the right set
    int setIdx = -1;
    unsigned int bb = getLogicBB(op.first).value();
    int sccIdx = scc[bb];
    if (sccIdx == -1) {
      // Operation not part of a set
      opT.opsNotOnCfg.push_back(op.first);
      continue;
    }
    auto setSelect = opT.setSelect.find(sccIdx);
    if (setSelect != opT.setSelect.end()) {
      setIdx = setSelect->second;
    } else {
      setIdx = opT.setSelect.size();
      opT.setSelect[sccIdx] = setIdx;
      opT.sets.emplace_back(sccIdx, latency);
    }
    Set &setT = opT.sets[setIdx];

    // Simply add group to set
    setT.groups.push_front(groupItem);
  }
}

int ResourceSharing::getNumberOfBasicBlocks() {
  unsigned int maximum = 0;
  for (auto archItem : archs) {
    maximum = std::max(maximum, std::max(archItem.srcBB, archItem.dstBB));
  }
  return maximum + 1; // as we have BB0, we need to add one at the end
}

void ResourceSharing::getListOfControlFlowEdges(
    SmallVector<dynamatic::experimental::ArchBB> archsExt) {
  archs = std::move(archsExt);
}

std::vector<int> ResourceSharing::performSccBbl() {
  return kosarajusAlgorithmBbl(archs);
}

void ResourceSharing::performSccOpl(std::set<mlir::Operation *> &result,
                                     handshake::FuncOp *funcOp) {
  kosarajusAlgorithmOpl(firstOp, funcOp, result);
}

void ResourceSharing::print() {
  llvm::errs() << "\n***** Basic Blocks *****\n";
  for (auto archItem : archs) {
    llvm::errs() << "Source: " << archItem.srcBB
                 << ", Destination: " << archItem.dstBB << "\n";
  }
  std::map<int, double>::iterator it = throughput.begin();
  llvm::errs() << "\n**** Throughput per CFDFC ****\n";
  for (; it != throughput.end(); it++) {
    llvm::errs() << "CFDFC #" << it->first << ": " << it->second << "\n";
  }
  for (const auto& op : operationTypes) {
    llvm::errs() << "\n*** New Operation type: " << op.identifier << " ***\n";
    for (const auto& set : op.sets) {
      llvm::errs() << "** New set **\n";
      for (const auto& group : set.groups) {
        llvm::errs() << "* New group *\n";
        llvm::errs() << "Number of entries: " << group.items.size() << "\n";
      }
    }
  }
}

void ResourceSharing::getControlStructure(handshake::FuncOp funcOp) {
  ControlStructure controlItem;
  unsigned int bbIdx = 0;
  for (Operation &op : funcOp.getOps()) {
    if (op.getName().getStringRef() == "handshake.merge" ||
        op.getName().getStringRef() == "handshake.control_merge") {
      for (const auto &u : op.getResults()) {
        if (u.getType().isa<NoneType>()) {
          bbIdx = getLogicBB(&op).value();
          controlItem.controlMerge = u;
        }
      }
    }
    if (op.getName().getStringRef() == "handshake.br" ||
        op.getName().getStringRef() == "handshake.cond_br") {
      for (const auto &u : op.getOperands()) {
        if (u.getType().isa<NoneType>()) {
          if (bbIdx != getLogicBB(&op).value()) {
            llvm::errs() << "[critical Error] control channel not present\n";
          }
          controlItem.controlBranch = u;
          controlMap[bbIdx] = controlItem;
        }
      }
    }
  }
}

void ResourceSharing::placeAndComputeNecessaryDataFromPerformanceAnalysis(
    ResourceSharingInfo data, const TimingDatabase& timingDB) {
  // comput first operation of the IR
  computeFirstOp(data.funcOp);

  // initialize topological sorting to determine topological order
  initializeTopolocialOpSort(&data.funcOp);

  // find non-cyclic operations
  std::set<mlir::Operation *> opsWithNoLoops;
  performSccOpl(opsWithNoLoops, &data.funcOp);

  // get the connections between basic blocks
  getListOfControlFlowEdges(data.archs);

  // perform SCC computation
  std::vector<int> scc = performSccBbl();

  // get number of strongly connected components
  int numberOfScc = scc.size();

  // fill resource sharing class with all shareable operations
  retrieveDataFromPerformanceAnalysis(data, scc, numberOfScc, timingDB);

  // get control structures of each BB
  getControlStructure(data.funcOp);
}

/*
 * Extending FPGA20Buffers class
 */

std::vector<ResourceSharingInfo::OperationData> MyFPGA20Buffers::getData() {
  std::vector<ResourceSharingInfo::OperationData> returnInfo;
  ResourceSharingInfo::OperationData sharingItem;
  double throughput, latency;
  for (auto [idx, cfdfcWithVars] : llvm::enumerate(vars.cfVars)) {
    auto [cf, cfVars] = cfdfcWithVars;
    throughput = cfVars.throughput.get(GRB_DoubleAttr_X);

    for (auto &[op, unitVars] : cfVars.unitVars) {
      sharingItem.op = op;
      if (failed(timingDB.getLatency(op, SignalType::DATA, latency)) ||
          latency == 0.0)
        continue;
      sharingItem.occupancy = latency * throughput;
      returnInfo.push_back(sharingItem);
    }
  }
  return returnInfo;
}

double MyFPGA20Buffers::getOccupancySum(std::set<Operation *> &group) {
  std::map<Operation *, double> occupancies;
  for (auto *item : group) {
    occupancies[item] = -1.0;
  }
  double throughput, latency, occupancy;
  for (auto [idx, cfdfcWithVars] : llvm::enumerate(vars.cfVars)) {
    auto [cf, cfVars] = cfdfcWithVars;
    // for each CFDFC, extract the throughput in double format
    throughput = cfVars.throughput.get(GRB_DoubleAttr_X);

    for (auto &[op, unitVars] : cfVars.unitVars) {
      if (group.find(op) != group.end()) {
        if (failed(timingDB.getLatency(op, SignalType::DATA, latency)) ||
            latency == 0.0)
          continue;
        occupancy = latency * throughput;
        occupancies[op] = std::max(occupancy, occupancies[op]);
      }
    }
  }
  double sum = 0.0;
  for (auto item : occupancies) {
    //assert(item.second > 0 && "Incorrect occupancy\n");
    sum += item.second;
  }
  return sum;
}

LogicalResult
MyFPGA20Buffers::addSyncConstraints(const std::vector<Value>& opaqueChannel) {
  for (auto channel : opaqueChannel) {
    ChannelVars &chVars = vars.channelVars[channel];
    auto dataIt = chVars.signalVars.find(SignalType::DATA);
    GRBVar &dataOpaque = dataIt->second.bufPresent;
    GRBVar &opaque = chVars.bufPresent;
    model.addConstr(opaque == 1.0, "additional_opaque_channel");
    model.addConstr(dataOpaque == 1.0, "additional_opaque_channel");
  }
  return success();
}

/*
 *   indroducing version of next_permutation
 */
namespace permutation {

void findBBEdges(std::deque<std::pair<int, int>> &bbOps,
                 std::vector<Operation *> &permutationVector) {
  std::sort(permutationVector.begin(), permutationVector.end(),
            [](Operation *a, Operation *b) -> bool {
              return (getLogicBB(a) < getLogicBB(b)) ||
                     (getLogicBB(a) == getLogicBB(b) && (a < b));
            });
  int size = permutationVector.size();
  int start, end = 0;
  while (end != size) {
    start = end;
    unsigned int basicBlockId = getLogicBB(permutationVector[start]).value();
    while (end != size &&
           getLogicBB(permutationVector[end]).value() == basicBlockId) {
      ++end;
    }
    bbOps.emplace_front(start, end);
  }
}

bool getNextPermutation(PermutationEdge beginOfPermutationVector,
                          std::deque<std::pair<int, int>> &separationOfBBs) {
  for(auto [start, end] : separationOfBBs) {
    if (next_permutation(beginOfPermutationVector + start,
                         beginOfPermutationVector + end)) {
      return true;
    }
  }
  return false;
}

} // namespace permutation

namespace dynamatic {
namespace experimental {
namespace sharing {

/*
 * additional functions used for resource sharing
 */

bool lessOrEqual(double a, double b) {
  double diff = 0.000001;
  return a < b + diff;
}

bool equal(double a, double b) {
  double diff = 0.000001;
  return (a + diff > b) && (b + diff > a);
}

std::vector<std::pair<GroupIt, GroupIt>>
combinations(Set *set, std::map<Group, std::set<Group>> &alreadyTested) {
  std::vector<std::pair<GroupIt, GroupIt>> result;
  for (GroupIt g1 = set->groups.begin(); g1 != set->groups.end(); g1++) {
    std::set<Group> testedGroups = alreadyTested[*g1];
    GroupIt g2 = g1;
    g2++;
    for (; g2 != set->groups.end(); g2++) {
      if (testedGroups.find(*g2) != testedGroups.end()) {
        // already tested
        continue;
      }
      result.emplace_back(g1, g2);
    }
  }
  return result;
}

void revertToInitialState(std::map<int, ControlStructure> &controlMap) {
  for (auto &item : controlMap) {
    item.second.currentPosition = item.second.controlMerge;
  }
}

Value generatePerformanceStep(OpBuilder *builder, mlir::Operation *op,
                                std::map<int, ControlStructure> &controlMap) {
  Value returnValue;
  mlir::Value controlMerge =
      controlMap[getLogicBB(op).value()].currentPosition;
  builder->setInsertionPointAfterValue(controlMerge);
  // child operation of control merge
  mlir::Operation *childOp = controlMerge.getUses().begin()->getOwner();
  // get control operands of newly created syncOp
  std::vector<Value> controlOperands = {controlMerge};
  for (auto value : op->getOperands()) {
    controlOperands.push_back(value);
  }
  // create syncOp
  mlir::Operation *syncOp =
      builder->create<SyncOp>(controlMerge.getLoc(), controlOperands);
  returnValue = syncOp->getResult(0);
  // connect syncOp
  childOp->replaceUsesOfWith(controlMerge, syncOp->getResult(0));
  int controlInt = 0;
  for (auto value : op->getOperands()) {
    op->replaceUsesOfWith(value, syncOp->getResult(controlInt + 1));
    controlInt++;
  }
  controlMap[getLogicBB(op).value()].currentPosition = syncOp->getResult(0);
  inheritBB(op, syncOp);
  return returnValue;
}

std::vector<Value>
generatePerformanceModel(OpBuilder *builder,
                           std::vector<mlir::Operation *> &items,
                           std::map<int, ControlStructure> &controlMap) {
  std::vector<Value> returnValues;
  revertToInitialState(controlMap);
  returnValues.reserve(items.size());
for (auto *op : items) {
    returnValues.push_back(
        generatePerformanceStep(builder, op, controlMap));
  }
  return returnValues;
}

void revertPerformanceStep(OpBuilder *builder, mlir::Operation *op) {
  // get specific syncOp
  mlir::Operation *syncOp = op->getOperand(0).getDefiningOp();
  // reconnect the previous state
  int controlInt = 0;
  for (auto value : syncOp->getResults()) {
    value.replaceAllUsesWith(syncOp->getOperand(controlInt));
    controlInt++;
  }
  // delete syncOp
  syncOp->erase();
}

void destroyPerformanceModel(OpBuilder *builder,
                               std::vector<mlir::Operation *> &items) {
  for (auto *op : items) {
    revertPerformanceStep(builder, op);
  }
}

// handshake::ForkOp
mlir::OpResult extendFork(OpBuilder *builder, ForkOp oldFork) {
  Operation *opSrc = oldFork.getOperand().getDefiningOp();
  Value opSrcIn = oldFork.getOperand();
  std::vector<Operation *> opsToProcess;
  for (auto &u : oldFork.getResults().getUses())
    opsToProcess.push_back(u.getOwner());

  // Insert fork after op
  builder->setInsertionPointAfter(opSrc);
  auto forkSize = opsToProcess.size();

  auto newForkOp =
      builder->create<ForkOp>(opSrcIn.getLoc(), opSrcIn, forkSize + 1);
  inheritBB(opSrc, newForkOp);
  for (int i = 0, e = forkSize; i < e; ++i)
    opsToProcess[i]->replaceUsesOfWith(oldFork->getResult(i),
                                       newForkOp->getResult(i));
  oldFork.erase();
  return newForkOp->getResult(forkSize);
}

// handshake::SinkOp
void addSink(OpBuilder *builder, mlir::OpResult *connectionPoint) {
  builder->setInsertionPointAfter(
      connectionPoint->getOwner()->getOperand(0).getDefiningOp());
  builder->create<SinkOp>(connectionPoint->getLoc(), *connectionPoint);
}

// handshake::ConstantOp
mlir::OpResult addConst(OpBuilder *builder, mlir::OpResult *connectionPoint,
                        int value) {
  Operation *opSrc = connectionPoint->getOwner()->getOperand(0).getDefiningOp();
  builder->setInsertionPointAfter(opSrc);
  IntegerAttr cond = builder->getBoolAttr(value);
  auto newConstOp = builder->create<handshake::ConstantOp>(
      connectionPoint->getLoc(), cond.getType(), cond, *connectionPoint);
  inheritBB(opSrc, newConstOp);
  return newConstOp->getResult(0);
}

// handshake::BranchOp
mlir::OpResult addBranch(OpBuilder *builder,
                         mlir::OpResult *connectionPoint) { // Value
  Operation *opSrc = connectionPoint->getOwner()->getOperand(0).getDefiningOp();
  auto newBranchOp = builder->create<handshake::BranchOp>(
      connectionPoint->getLoc(), *connectionPoint);
  inheritBB(opSrc, newBranchOp);
  return newBranchOp->getResult(0);
}

void deleteAllBuffers(FuncOp funcOp) {
  for (auto op : llvm::make_early_inc_range(funcOp.getOps<TEHBOp>())) {
    op->getResult(0).replaceAllUsesWith(op->getOperand(0));
    // delete buffer
    op->erase();
  }
  for (auto op : llvm::make_early_inc_range(funcOp.getOps<OEHBOp>())) {
    op->getResult(0).replaceAllUsesWith(op->getOperand(0));
    // delete buffer
    op->erase();
  }
}

Group::~Group(){};
} // namespace sharing
} // namespace experimental
} // namespace dynamatic
