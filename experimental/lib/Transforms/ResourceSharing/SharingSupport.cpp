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
#include "experimental/Transforms/ResourceSharing/SharingSupport.h"

using namespace dynamatic::handshake;
using namespace dynamatic::buffer::fpga20;
using namespace dynamatic::experimental::sharing;

void ResourceSharingInfo::OperationData::print() {
    llvm::errs() << "Operation " << op
                << ", occupancy: " << occupancy
                << ", block: " << getLogicBB(op)
                << "\n";
}

void ResourceSharingInfo::computeOccupancySum() {
    return;
}

void Group::addOperation(mlir::Operation* op) {
    items.push_back(op);
}

bool Group::recursivelyDetermineIfCyclic(mlir::Operation* current_op, std::set<mlir::Operation*>& node_visited, mlir::Operation* op) {
    node_visited.insert(current_op);
    for (auto &u : current_op->getResults().getUses()) {
        Operation *child_op = u.getOwner();
        if(child_op == op) {
            return true;
        }
        auto it = node_visited.find(child_op);
        if(it == node_visited.end()) {
            //not visited yet
            if(recursivelyDetermineIfCyclic(child_op, node_visited, op)) {
                return true;
            }
        }
    }
    return false;
}

bool Group::determineIfCyclic(mlir::Operation* op) {
    std::set<mlir::Operation*> node_visited;
    return recursivelyDetermineIfCyclic(op, node_visited, op);
}

void Set::addGroup(Group group) {
    groups.push_back(group);
}

void Set::joinGroups(GroupIt group1, GroupIt group2, std::vector<mlir::Operation*>& finalOrd) {
    Group newly_created = Group(finalOrd, group1->shared_occupancy + group1->shared_occupancy, group1->hasCycle | group2->hasCycle);
    groups.erase(group1);
    groups.erase(group2);
    groups.push_back(newly_created);
}

void Set::joinSet(Set *joined_element) {
    GroupIt pelem = groups.begin();
    for(GroupIt jelem = joined_element->groups.begin(); jelem != joined_element->groups.end(); pelem++, jelem++) {
        pelem->items.insert(pelem->items.end(),
                            jelem->items.begin(),
                            jelem->items.end()
                            );
    }
}
// std::vector<mlir::Operation*> items;
void Set::print() {
    llvm::errs() << "Set id: " << SCC_id << "\n";
    int i = 0;
    for(auto group : groups) {
        llvm::errs() << "Group #" << i++ << "\n";
        for(auto op : group.items) {
            llvm::errs() << op << ", ";
        }
        llvm::errs() << "\n";
    }
}

void ResourceSharingForSingleType::addSet(Group group) {
    sets.push_back(Set(group));
}

void ResourceSharingForSingleType::print() {
    llvm::errs() << identifier << "\n";
    for(auto set : sets) {
        llvm::errs() << "SCC"  << set.SCC_id << ":\n";
        int group_count = 0;
        for(auto group : set.groups) {
            llvm::errs() << "Group " << group_count++ << ": ";
            for(auto item : group.items) {
                llvm::errs() << item << ", ";
            }
        }
        llvm::errs() << "\n";
    }
}

void ResourceSharingForSingleType::printFinalGroup() {
    llvm::errs() << "Final grouping for " <<identifier << ":\n";
    int group_count = 0;
    for(auto group : final_grouping.groups) {
        llvm::errs() << "Group " << group_count++ << ": ";
        for(auto item : group.items) {
            llvm::errs() << item << ", ";
        }
    }
    llvm::errs() << "\n";
}

void ResourceSharingForSingleType::sharingAcrossLoopNests() {
    int number_of_sets = sets.size();
    if(!number_of_sets) {
        return;
    }

    int max_set_size = -1;
    int max_idx = -1;
    for(int i = 0; i < number_of_sets; i++) {
        if((int)sets[i].groups.size() > max_set_size) {
            max_set_size = sets[i].groups.size();
            max_idx = i;
        }
    }
    //choose initial set
    final_grouping = sets[max_idx];

    for(int i = 0; i < number_of_sets; i++) {
        if(i == max_idx) {
            continue;
        }
        final_grouping.joinSet(&sets[i]);
    }
}

void ResourceSharingForSingleType::sharingOtherUnits() {
    auto it = final_grouping.groups.begin();
    for(auto unit : Ops_not_on_CFG) {
        it->addOperation(unit);
        it++;
        if(it == final_grouping.groups.end()) {
            it = final_grouping.groups.begin();
        }
    }
}

void ResourceSharing::recursiveDFStravel(Operation *op, unsigned int *position, std::set<mlir::Operation*>& node_visited) {
    //add operation
    node_visited.insert(op);

    //DFS over all child ops
    for (auto &u : op->getResults().getUses()) {
        Operation *child_op = u.getOwner();
        auto it = node_visited.find(child_op);
        if(it == node_visited.end()) {
            //not visited yet
            recursiveDFStravel(child_op, position, node_visited);
        }
    }
    //update container
    OpTopologicalOrder[op] = *position;
    ++(*position);
    return;
}

bool ResourceSharing::computeFirstOp(handshake::FuncOp funcOp) {
    // If we are in the entry block, we can use the start input of the
    // function (last argument) as our control value
    if(!funcOp.getArguments().back().getType().isa<NoneType>()) {
        return false;
    }
    Value func = funcOp.getArguments().back();
    std::vector<Operation *> startingOps;
    for (auto &u : func.getUses())
        startingOps.push_back(u.getOwner());
    if(startingOps.size() != 1) {
        return false;
    }
    firstOp = startingOps[0];
    return true;
}

Operation *ResourceSharing::getFirstOp() {
    return firstOp;
}

void ResourceSharing::initializeTopolocialOpSort(handshake::FuncOp *funcOp) {
    if(firstOp == nullptr) {
        llvm::errs() << "[Error] Operation directly after start not yet present\n";
    }
    unsigned int position = 0;
    std::set<mlir::Operation*> node_visited;
    recursiveDFStravel(firstOp, &position, node_visited);
    for (Operation &op : funcOp->getOps()) {
        auto it = node_visited.find(&op);
        if(it == node_visited.end()) {
            recursiveDFStravel(&op, &position, node_visited);
        }
    }
    return;
}

void ResourceSharing::printTopologicalOrder() {
    llvm::errs() << "Topological Order: \n";
    for(auto [op, id] : OpTopologicalOrder) {
        llvm::errs() << id << " : " << op << "\n";
    }
}

std::vector<Operation*> ResourceSharing::sortTopologically(GroupIt group1, GroupIt group2) {
    std::vector<Operation*> result(group1->items.size() + group2->items.size());
    //add all operations in sorted order
    merge(group1->items.begin(), group1->items.end(), group2->items.begin(), group2->items.end(), result.begin(), [this](Operation *a, Operation *b) {return OpTopologicalOrder[a] > OpTopologicalOrder[b];});
    return result;
}

bool ResourceSharing::isTopologicallySorted(std::vector<Operation*> Ops) {
    for(unsigned long i = 0; i < Ops.size() - 1; i++) {
        if(OpTopologicalOrder[Ops[i]] > OpTopologicalOrder[Ops[i+1]]) {
            return false;
        }
    }
    return true;
}

void ResourceSharing::retrieveDataFromPerformanceAnalysis(ResourceSharingInfo sharing_feedback, std::vector<int>& SCC, int number_of_SCC, TimingDatabase timingDB) {
    // take biggest occupancy per operation
    std::unordered_map<mlir::Operation*, double> uniqueOperation;
    for(auto item : sharing_feedback.operations) {
        if (uniqueOperation.find(item.op) != uniqueOperation.end()) {
            // operation already present
            uniqueOperation[item.op] = std::max(item.occupancy, uniqueOperation[item.op]);
        } else {
            // add operation
            uniqueOperation[item.op] = item.occupancy;
        }
    }

    //everytime we place/overwrite data, initial number of operation types is 0;
    number_of_operation_types = 0;

    //iterate through all retrieved operations
    for(auto op : uniqueOperation) {
        //choose the right operation type
        double latency;
        if (failed(timingDB.getLatency(op.first, SignalType::DATA, latency)))
            latency = 0.0;

        llvm::StringRef OpName = op.first->getName().getStringRef();
        Group group_item = Group(op.first, op.second);
        int OpIdx = -1;
        auto item = OpNames.find(OpName);
        if(item != OpNames.end()) {
            OpIdx = item->second;
        } else {
            OpNames[OpName] = number_of_operation_types;
            OpIdx = number_of_operation_types;
            ++number_of_operation_types;
            operationTypes.push_back(ResourceSharingForSingleType(latency, OpName));
        }
        ResourceSharingForSingleType& OpT = operationTypes[OpIdx];

        //choose the right set
        int SetIdx = -1;
        unsigned int BB = getLogicBB(op.first).value();
        int SCC_idx = SCC[BB];
        if(SCC_idx == -1) {
            //Operation not part of a set
            OpT.Ops_not_on_CFG.push_back(op.first);
            continue;
        }
        auto set_select = OpT.SetSelect.find(SCC_idx);
        if(set_select != OpT.SetSelect.end()) {
            SetIdx = set_select->second;
        } else {
            SetIdx = OpT.SetSelect.size();
            OpT.SetSelect[SCC_idx] = SetIdx;
            OpT.sets.push_back(Set(SCC_idx, latency));
        }
        Set& SetT = OpT.sets[SetIdx];

        //Simply add group to set
        SetT.groups.push_front(group_item);
    }
}

int ResourceSharing::getNumberOfBasicBlocks() {
    unsigned int maximum = 0;
    for(auto arch_item : archs) {
        maximum = std::max(maximum, std::max(arch_item.srcBB, arch_item.dstBB));
    }
    return maximum + 1; //as we have BB0, we need to add one at the end
}

void ResourceSharing::getListOfControlFlowEdges(SmallVector<dynamatic::experimental::ArchBB> archs_ext) {
    archs = archs_ext;
}

std::vector<int> ResourceSharing::performSCC_bbl() {
    return Kosarajus_algorithm_BBL(archs);
}

void ResourceSharing::performSCC_opl(std::set<mlir::Operation*>& result, handshake::FuncOp *funcOp) {
    Kosarajus_algorithm_OPL(firstOp, funcOp, result);
}

void ResourceSharing::print() {
    llvm::errs() << "\n***** Basic Blocks *****\n";
    for(auto arch_item : archs) {
        llvm::errs() << "Source: " << arch_item.srcBB << ", Destination: " << arch_item.dstBB << "\n";
    }
    std::map<int, double>::iterator it = throughput.begin();
    llvm::errs() << "\n**** Throughput per CFDFC ****\n";
    for(; it != throughput.end(); it++) {
        llvm::errs() << "CFDFC #" << it->first << ": " << it->second << "\n";
    }
    for(auto Op : operationTypes) {
        llvm::errs() << "\n*** New Operation type: " << Op.identifier << " ***\n";
        for(auto set : Op.sets) {
            llvm::errs() << "** New set **\n";
            for(auto group : set.groups) {
                llvm::errs() << "* New group *\n";
                llvm::errs() << "Number of entries: " << group.items.size() << "\n";
            }
        }
    }
}

void ResourceSharing::getControlStructure(handshake::FuncOp funcOp) {
    controlStructure control_item;
    unsigned int BB_idx = 0;
    for (Operation &op : funcOp.getOps()) {
        if(op.getName().getStringRef() == "handshake.merge" || op.getName().getStringRef() == "handshake.control_merge") {
            for (const auto &u : op.getResults()) {
                if(u.getType().isa<NoneType>()) {
                    BB_idx = getLogicBB(&op).value();
                    control_item.control_merge = u;
                }
            }
        }
        if(op.getName().getStringRef() == "handshake.br" || op.getName().getStringRef() == "handshake.cond_br") {
            for (const auto &u : op.getOperands()) {
                if(u.getType().isa<NoneType>()) {
                    if(BB_idx != getLogicBB(&op).value()) {
                        llvm::errs() << "[critical Error] control channel not present\n";
                    }
                    control_item.control_branch = u;
                    control_map[BB_idx] = control_item;
                }
            }
        }
    }
    return;
}

void ResourceSharing::placeAndComputeNecessaryDataFromPerformanceAnalysis(ResourceSharingInfo data, TimingDatabase timingDB) {
    // comput first operation of the IR
    computeFirstOp(data.funcOp);

    // initialize topological sorting to determine topological order
    initializeTopolocialOpSort(&data.funcOp);
    
    // find non-cyclic operations
    std::set<mlir::Operation*> ops_with_no_loops;
    performSCC_opl(ops_with_no_loops, &data.funcOp);
    
    // get the connections between basic blocks
    getListOfControlFlowEdges(data.archs);

    // perform SCC computation
    std::vector<int> SCC = performSCC_bbl();

    // get number of strongly connected components
    int number_of_SCC = SCC.size();
    
    // fill resource sharing class with all shareable operations
    retrieveDataFromPerformanceAnalysis(data, SCC, number_of_SCC, timingDB);
    
    // get control structures of each BB
    getControlStructure(data.funcOp);
}

/*
 * Extending FPGA20Buffers class
 */

std::vector<ResourceSharingInfo::OperationData> MyFPGA20Buffers::getData() {
    std::vector<ResourceSharingInfo::OperationData> return_info;
    ResourceSharingInfo::OperationData sharing_item;
    double throughput, latency;
    for (auto [idx, cfdfcWithVars] : llvm::enumerate(vars.cfVars)) {
        auto [cf, cfVars] = cfdfcWithVars;
        throughput = cfVars.throughput.get(GRB_DoubleAttr_X);

        for (auto &[op, unitVars] : cfVars.unitVars) {
        sharing_item.op = op;
        if (failed(timingDB.getLatency(op, SignalType::DATA, latency)) || latency == 0.0)
            continue;
        sharing_item.occupancy = latency * throughput;
        return_info.push_back(sharing_item);
        }
    }
    return return_info;
}

double MyFPGA20Buffers::getOccupancySum(std::set<Operation*>& group) {
    std::map<Operation*, double> occupancies;
    for(auto item : group) {
        occupancies[item] = -1.0;
    }
    double throughput, latency, occupancy;
    for (auto [idx, cfdfcWithVars] : llvm::enumerate(vars.cfVars)) {
        auto [cf, cfVars] = cfdfcWithVars;
        // for each CFDFC, extract the throughput in double format
        throughput = cfVars.throughput.get(GRB_DoubleAttr_X);

        for (auto &[op, unitVars] : cfVars.unitVars) {
            if(group.find(op) != group.end()) {
                if (failed(timingDB.getLatency(op, SignalType::DATA, latency)) || latency == 0.0)
                    continue;
                occupancy = latency * throughput;
                occupancies[op] = std::max(occupancy, occupancies[op]);
            }
        }
    }
    double sum = 0.0;
    for(auto item : occupancies) {
        assert(item.second > 0 && "Incorrect occupancy\n");
        sum += item.second;
    }
    return sum;
}

LogicalResult MyFPGA20Buffers::addSyncConstraints(std::vector<Value> opaqueChannel) {
    for(auto channel : opaqueChannel) {
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

void findBBEdges(std::deque<std::pair<int, int>>& BBops, std::vector<Operation*>& permutation_vector) {
    std::sort(permutation_vector.begin(), permutation_vector.end(), [](Operation *a, Operation *b) -> bool {return (getLogicBB(a) < getLogicBB(b)) || (getLogicBB(a) == getLogicBB(b) && (a < b));});
    int size = permutation_vector.size();
    int start, end = 0;
    while(end != size) {
        start = end;
        unsigned int BasicBlockId = getLogicBB(permutation_vector[start]).value();
        while(end != size && getLogicBB(permutation_vector[end]).value() == BasicBlockId) {
            ++end;
        }
        BBops.push_front(std::make_pair(start, end));
    }
}

bool get_next_permutation(PermutationEdge begin_of_permutation_vector, std::deque<std::pair<int, int>>& separation_of_BBs) {
    for(auto [start, end] : separation_of_BBs) {
        if(next_permutation (begin_of_permutation_vector + start, begin_of_permutation_vector + end)) {
          return true;
        }
    }
    return false;
}

} //namespace permutation

namespace dynamatic {
namespace experimental {
namespace sharing {

/*
 * additional functions used for resource sharing
 */

bool lessOrEqual(double a, double b) {
    double diff = 0.000001;
    if((a < b + diff)) {
        return true;
    }
    return false;
}

bool equal(double a, double b) {
    double diff = 0.000001;
    if((a + diff > b)  && (b + diff > a)) {
        return true;
    }
    return false;
}

std::vector<std::pair<GroupIt, GroupIt>> combinations(Set *set, std::map<Group, std::set<Group>>& alreadyTested) {
    std::vector<std::pair<GroupIt, GroupIt>> result;
    for(GroupIt g1 = set->groups.begin(); g1 != set->groups.end(); g1++) {
        std::set<Group> tested_groups = alreadyTested[*g1];
        GroupIt g2 = g1;
        g2++;
        for( ; g2 != set->groups.end(); g2++) {
            if(alreadyTested.find(*g2) != alreadyTested.end()) {
                // already tested
                continue;
            }
            result.push_back(std::make_pair(g1, g2));
        }
    }
    return result;
}


void revert_to_initial_state(std::map<int, controlStructure>& control_map) {
  for(auto& item : control_map) {
    item.second.current_position = item.second.control_merge;
  }
}

Value generate_performance_step(OpBuilder* builder, mlir::Operation *op, std::map<int, controlStructure>& control_map) {
  Value return_value;
  mlir::Value control_merge = control_map[getLogicBB(op).value()].current_position;
  builder->setInsertionPointAfterValue(control_merge);
  //child operation of control merge
  mlir::Operation *child_op = control_merge.getUses().begin()->getOwner();
  //get control operands of newly created syncOp
  std::vector<Value> controlOperands = {control_merge};
  for(auto value : op->getOperands()) {
    controlOperands.push_back(value);
  }
  //create syncOp
  mlir::Operation *syncOp = builder->create<SyncOp>(control_merge.getLoc(), controlOperands);
  return_value = syncOp->getResult(0);
  //connect syncOp
  child_op->replaceUsesOfWith(control_merge, syncOp->getResult(0));
  int control_int = 0;
  for(auto value : op->getOperands()) {
    op->replaceUsesOfWith(value, syncOp->getResult(control_int+1));
    control_int++;
  }
  control_map[getLogicBB(op).value()].current_position = syncOp->getResult(0);
  inheritBB(op, syncOp);
  return return_value;
}

std::vector<Value> generate_performance_model(OpBuilder* builder, std::vector<mlir::Operation*>& items, std::map<int, controlStructure>& control_map) {
  std::vector<Value> return_values;
  revert_to_initial_state(control_map);
  for(auto op : items) {
    return_values.push_back(generate_performance_step(builder, op,control_map));
  }
  return return_values;
}

void revert_performance_step(OpBuilder* builder, mlir::Operation *op) {
  //get specific syncOp
  mlir::Operation *syncOp = op->getOperand(0).getDefiningOp();
  //reconnect the previous state
  int control_int = 0;
  for(auto value : syncOp->getResults()) {
    value.replaceAllUsesWith(syncOp->getOperand(control_int));
    control_int++;
  }
  //delete syncOp
  syncOp->erase();
}

void destroy_performance_model(OpBuilder* builder, std::vector<mlir::Operation*>& items) {
  for(auto op : items) {
    revert_performance_step(builder, op);
  }
}


//handshake::ForkOp
mlir::OpResult extend_fork(OpBuilder* builder, ForkOp OldFork) {
      Operation *opSrc = OldFork.getOperand().getDefiningOp();
      Value opSrcIn = OldFork.getOperand();
      std::vector<Operation *> opsToProcess;
      for (auto &u : OldFork.getResults().getUses())
        opsToProcess.push_back(u.getOwner());

      // Insert fork after op
      builder->setInsertionPointAfter(opSrc);
      auto forkSize = opsToProcess.size();

      auto newForkOp = builder->create<ForkOp>(opSrcIn.getLoc(), opSrcIn, forkSize + 1);
      inheritBB(opSrc, newForkOp);
      for (int i = 0, e = forkSize; i < e; ++i)
        opsToProcess[i]->replaceUsesOfWith(OldFork->getResult(i), newForkOp->getResult(i));
      OldFork.erase();
      return newForkOp->getResult(forkSize);
}

//handshake::SinkOp
void addSink(OpBuilder* builder, mlir::OpResult* connectionPoint) {
    builder->setInsertionPointAfter(connectionPoint->getOwner()->getOperand(0).getDefiningOp());
    builder->create<SinkOp>(connectionPoint->getLoc(), *connectionPoint);
}

//handshake::ConstantOp
mlir::OpResult addConst(OpBuilder* builder, mlir::OpResult* connectionPoint, int value) {
    Operation *opSrc = connectionPoint->getOwner()->getOperand(0).getDefiningOp();
    builder->setInsertionPointAfter(opSrc);
    IntegerAttr cond = builder->getBoolAttr(value);
    auto newConstOp = builder->create<handshake::ConstantOp>(connectionPoint->getLoc(), cond.getType(), cond, *connectionPoint);
    inheritBB(opSrc, newConstOp);
    return newConstOp->getResult(0);
}

//handshake::BranchOp
mlir::OpResult addBranch(OpBuilder* builder, mlir::OpResult* connectionPoint) {  //Value
    Operation *opSrc = connectionPoint->getOwner()->getOperand(0).getDefiningOp();
    auto newBranchOp = builder->create<handshake::BranchOp>(connectionPoint->getLoc(), *connectionPoint);
    inheritBB(opSrc, newBranchOp);
    return newBranchOp->getResult(0);
}

void deleteAllBuffers(FuncOp funcOp) {
  for (auto op : llvm::make_early_inc_range(funcOp.getOps<TEHBOp>())) {
    op->getResult(0).replaceAllUsesWith(op->getOperand(0));
    //delete buffer
    op->erase();
  }
  for (auto op : llvm::make_early_inc_range(funcOp.getOps<OEHBOp>())) {
    op->getResult(0).replaceAllUsesWith(op->getOperand(0));
    //delete buffer
    op->erase();
  }
}

} // namespace sharing
} // namespace experimental
} // namespace dynamatic

