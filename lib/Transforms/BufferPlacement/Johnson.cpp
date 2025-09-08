#include "dynamatic/Transforms/BufferPlacement/Johnson.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/Logging.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Casting.h"
#include <string>

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using Cycle = std::vector<Operation *>;
using CycleList = std::vector<Cycle>;
using ChannelCycle = std::vector<Value>;
using ChannelCycleList = std::vector<ChannelCycle>;

GraphForJohnson::GraphForJohnson(FuncOp funcOp) {
  for (Operation &op : funcOp.getOps()) {
    for (auto result : op.getResults()) {
      for (auto &use : result.getUses()) {
        Operation *nextOp = use.getOwner();

        if (isa<handshake::MemoryControllerOp>(nextOp) ||
            isa<handshake::LSQOp>(nextOp))
          continue;

        adjacency[&op][nextOp] = result;
      }
    }
    vertices.push_back(&op);
    blocked[&op] = false;
    blockedMap[&op] = {};
    seenCycleHashes = {};
  }
}

ChannelCycleList GraphForJohnson::findAllChannelCycles() {
  ChannelCycleList channelCycles;

  findAllCycles();

  for (const auto &cycle : cycles) {
    ChannelCycle channelCycle;
    for (auto it = cycle.begin(), e = cycle.end(); it != e; ++it) {
      Operation *op = *it;
      Operation *nextOp = *(std::next(it) == e ? cycle.begin() : std::next(it));
      channelCycle.push_back(adjacency.lookup(op).lookup(nextOp));
    }
    channelCycles.push_back(channelCycle);
  }
  return channelCycles;
}

void GraphForJohnson::clearBlocked() {
  for (auto &[op, _] : blocked)
    blocked[op] = false;
  for (auto &[op, _] : blockedMap)
    blockedMap[op].clear();
}

void GraphForJohnson::unblock(Operation *u) {
  blocked[u] = false;
  for (auto opToUnblock : blockedMap[u]) {
    blockedMap[u].erase(opToUnblock);
    if (blocked[opToUnblock])
      unblock(opToUnblock);
  }
}

bool GraphForJohnson::findCyclesFromStart(Operation *currentOp,
                                          Operation *start) {

  bool foundCycle = false;
  stack.push_back(currentOp);
  blocked[currentOp] = true;

  for (auto &[op, _] : adjacency.lookup(currentOp)) {
    if (op == start) {
      // Found a cycle
      Cycle cycle(stack);
      addCycleToCycleList(cycle);
      foundCycle = true;
    } else if (!blocked[op]) {
      if (findCyclesFromStart(op, start))
        foundCycle = true;
    }
  }

  if (foundCycle) {
    unblock(currentOp);
  } else {
    for (auto &[op, _] : adjacency.lookup(currentOp))
      blockedMap[op].insert(currentOp);
  }

  stack.pop_back();
  return foundCycle;
}

std::vector<Cycle> GraphForJohnson::findAllCycles() {
  cycles.clear();
  for (Operation *s : vertices) {
    clearBlocked();
    findCyclesFromStart(s, s);
  }
  return cycles;
}

void GraphForJohnson::addCycleToCycleList(const Cycle &cycle) {
  Cycle normalized = normalizeCycle(cycle);
  std::string hash = hashCycle(normalized);

  if (!seenCycleHashes.contains(hash)) {
    seenCycleHashes.insert(hash);
    cycles.push_back(normalized);
    llvm::errs() << "Cycle found: " << hash << "\n";
  }
}

Cycle GraphForJohnson::normalizeCycle(const Cycle &cycle) {
  if (cycle.empty())
    return cycle;
  auto minIt = std::min_element(cycle.begin(), cycle.end());
  Cycle rotated;
  rotated.insert(rotated.end(), minIt, cycle.end());
  rotated.insert(rotated.end(), cycle.begin(), minIt);
  return rotated;
}

std::string GraphForJohnson::hashCycle(const Cycle &cycle) {
  std::string repr;
  llvm::raw_string_ostream rso(repr);

  for (auto *op : cycle) {
    rso << getUniqueName(op);
    rso << ";"; // separator between ops
  }
  rso.flush();

  return repr;
}

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED