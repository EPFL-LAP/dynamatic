//===- HandshakeSizeSelectDepth.cpp - Size select depth ---------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Sets the `antitokenDepth` attribute of each select inside a CFDFC to
// ceil(|latency_true - latency_false| / II), where II is the lowest II among
// the CFDFCs that contain the select and latency_x is the maximum latency from
// the closest fork that is an ancestor of both data inputs of the select to
// data input x. The depth bounds how many iterations the faster data input may
// run ahead of the slower one; a depth of 0 makes the select a join.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/CFG.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <optional>

// [START Boilerplate code for the MLIR pass]
#include "dynamatic/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
#define GEN_PASS_DEF_HANDSHAKESIZESELECTDEPTH
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

#define DEBUG_TYPE "handshake-size-select-depth"

using namespace mlir;
using namespace dynamatic;

/// Returns the latency (in cycles) an operation adds on the data path.
static int64_t getOpLatency(Operation *op) {
  if (auto bufferOp = dyn_cast<handshake::BufferOp>(op))
    return bufferOp.getLatencyDV();
  if (auto latencyOp = dyn_cast<handshake::LatencyInterface>(op)) {
    FailureOr<int64_t> latency = latencyOp.getLatency();
    if (succeeded(latency))
      return *latency;
  }
  return 0;
}

/// Returns whether the backward traversal stops at the operation. Merge-like
/// operations close the cycles of the CFDFC (e.g., loop headers), and memory
/// interfaces close the cycles between memory ports and the interface (e.g.,
/// load address -> memory controller -> load data).
static bool stopsTraversal(Operation *op) {
  return isa<handshake::MergeLikeOpInterface, handshake::MemoryOpInterface>(op);
}

/// Collects the forks that `val` is (transitively) produced from, with their
/// distance in number of operations.
static DenseMap<Operation *, unsigned> collectAncestorForks(Value val) {
  DenseMap<Operation *, unsigned> forks;
  DenseSet<Operation *> visited;
  std::deque<std::pair<Value, unsigned>> queue{{val, 0}};
  while (!queue.empty()) {
    auto [v, dist] = queue.front();
    queue.pop_front();
    Operation *defOp = v.getDefiningOp();
    if (!defOp || !visited.insert(defOp).second || stopsTraversal(defOp))
      continue;
    if (isa<handshake::ForkOp, handshake::LazyForkOp>(defOp))
      forks.try_emplace(defOp, dist);
    for (Value operand : defOp->getOperands())
      queue.emplace_back(operand, dist + 1);
  }
  return forks;
}

/// Returns the maximum latency from `forkOp` to `val`, or std::nullopt if `val`
/// is not produced from `forkOp`.
///
/// `memo` caches, for each operation already visited, the maximum latency from
/// `forkOp` to the operation's results (including the operation's own latency),
/// or std::nullopt if the operation is not reachable from `forkOp`. Paths from
/// the fork to the select often reconverge (e.g., at a join), so the cache
/// avoids walking the shared part of the paths again; it also lets the latency
/// of both data inputs of the select be computed with one cache. The cache is
/// only valid for one `forkOp`.
///
/// `onPath` holds the operations on the current recursion path. Every cycle of
/// the circuit goes through a merge-like operation or a memory interface, where
/// the traversal stops, so the recursion never reaches an operation that is
/// already on the path.
static std::optional<int64_t>
getMaxLatencyFrom(Value val, Operation *forkOp,
                  DenseMap<Operation *, std::optional<int64_t>> &memo,
                  DenseSet<Operation *> &onPath) {
  Operation *defOp = val.getDefiningOp();
  if (!defOp)
    return std::nullopt;
  if (defOp == forkOp)
    return 0;
  if (stopsTraversal(defOp))
    return std::nullopt;
  if (auto it = memo.find(defOp); it != memo.end())
    return it->second;
  [[maybe_unused]] bool notOnPath = onPath.insert(defOp).second;
  assert(notOnPath && "cycle that does not go through a merge-like operation "
                      "or a memory interface");

  std::optional<int64_t> maxLatency;
  for (Value operand : defOp->getOperands()) {
    if (std::optional<int64_t> latency =
            getMaxLatencyFrom(operand, forkOp, memo, onPath))
      maxLatency = std::max(maxLatency.value_or(0), *latency);
  }
  if (maxLatency)
    *maxLatency += getOpLatency(defOp);
  onPath.erase(defOp);
  memo[defOp] = maxLatency;
  return maxLatency;
}

/// Maps each basic block to the lowest II among the CFDFCs that contain it.
static DenseMap<unsigned, double> getLowestIIPerBB(handshake::FuncOp funcOp) {
  DenseMap<unsigned, double> lowestII;
  auto throughputAttr = getDialectAttr<handshake::CFDFCThroughputAttr>(funcOp);
  auto bbListAttr = getDialectAttr<handshake::CFDFCToBBListAttr>(funcOp);
  if (!throughputAttr || !bbListAttr)
    return lowestII;

  for (NamedAttribute cfdfc : bbListAttr.getCfdfcMap()) {
    auto throughput = dyn_cast_if_present<FloatAttr>(
        throughputAttr.getThroughputMap().get(cfdfc.getName()));
    if (!throughput || throughput.getValueAsDouble() <= 0)
      continue;
    double ii = 1.0 / throughput.getValueAsDouble();
    for (Attribute bb : cast<ArrayAttr>(cfdfc.getValue())) {
      unsigned bbIdx = cast<IntegerAttr>(bb).getUInt();
      auto [it, inserted] = lowestII.try_emplace(bbIdx, ii);
      if (!inserted)
        it->second = std::min(it->second, ii);
    }
  }
  return lowestII;
}

namespace {

struct HandshakeSizeSelectDepthPass
    : public dynamatic::impl::HandshakeSizeSelectDepthBase<
          HandshakeSizeSelectDepthPass> {

  void runOnOperation() override {
    for (auto funcOp : getOperation().getOps<handshake::FuncOp>())
      sizeSelects(funcOp);
  }

private:
  void sizeSelects(handshake::FuncOp funcOp);
};

} // namespace

void HandshakeSizeSelectDepthPass::sizeSelects(handshake::FuncOp funcOp) {
  DenseMap<unsigned, double> lowestII = getLowestIIPerBB(funcOp);
  MLIRContext *ctx = funcOp.getContext();

  funcOp.walk([&](handshake::SelectOp selectOp) {
    std::optional<unsigned> bb = getLogicBB(selectOp);
    auto iiIt = bb ? lowestII.find(*bb) : lowestII.end();
    if (iiIt == lowestII.end()) {
      LLVM_DEBUG(llvm::dbgs()
                 << getUniqueName(selectOp) << ": not in a CFDFC, unchanged\n");
      return;
    }
    double ii = iiIt->second;

    // Closest fork that is an ancestor of both data inputs
    DenseMap<Operation *, unsigned> trueForks =
        collectAncestorForks(selectOp.getTrueValue());
    DenseMap<Operation *, unsigned> falseForks =
        collectAncestorForks(selectOp.getFalseValue());
    Operation *forkOp = nullptr;
    unsigned forkDist = 0;
    for (auto [op, trueDist] : trueForks) {
      auto falseIt = falseForks.find(op);
      if (falseIt == falseForks.end())
        continue;
      unsigned dist = std::max(trueDist, falseIt->second);
      if (!forkOp || dist < forkDist) {
        forkOp = op;
        forkDist = dist;
      }
    }
    if (!forkOp) {
      LLVM_DEBUG(llvm::dbgs() << getUniqueName(selectOp)
                              << ": no common ancestor fork, unchanged\n");
      return;
    }

    // Cache of the maximum latency from `forkOp` to each visited operation,
    // shared by the two data inputs (see getMaxLatencyFrom)
    DenseMap<Operation *, std::optional<int64_t>> memo;
    DenseSet<Operation *> onPath;
    int64_t trueLatency =
        getMaxLatencyFrom(selectOp.getTrueValue(), forkOp, memo, onPath)
            .value_or(0);
    int64_t falseLatency =
        getMaxLatencyFrom(selectOp.getFalseValue(), forkOp, memo, onPath)
            .value_or(0);

    // The faster data input runs ahead of the slower one by the difference of
    // their latencies. Small tolerance so that, e.g., II = 1 / 0.1666... does
    // not round up. Equal latencies give a depth of 0 (join).
    int64_t latencyDiff = std::abs(trueLatency - falseLatency);
    int64_t depth = std::max<int64_t>(
        0, static_cast<int64_t>(std::ceil(latencyDiff / ii - 1e-6)));
    selectOp.setAntitokenDepthAttr(
        IntegerAttr::get(IntegerType::get(ctx, 64), depth));
    LLVM_DEBUG(llvm::dbgs() << getUniqueName(selectOp) << ": II = " << ii
                            << ", fork = " << getUniqueName(forkOp)
                            << ", true latency = " << trueLatency
                            << ", false latency = " << falseLatency
                            << ", antitokenDepth = " << depth << "\n");
  });
}
