//===- MemDepAnalysis.cpp ---------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Usage 1: Legacy Dynamatic's memory analysis based on Polly
///
/// ```
/// opt input.ll \
///   -load-pass-plugin "$DYNAMATIC_DIR/build/lib/MemDepAnalysis.so"
///   -passes="mem-dep-analysis" \
///   -polly-process-unprofitable \
///   > output.ll
/// ```
///
/// Usage 2: A implementation based on "llvm/Analysis/DependenceAnalysis.h". We
/// use this to supply constraints to SDC-based static scheduling. This option
/// is enabled by using the flag `-use-dependence-analysis`
///
/// ```
/// opt input.ll \
///   -load-pass-plugin "$DYNAMATIC_DIR/build/lib/MemDepAnalysis.so"
///   -passes="mem-dep-analysis" \
///   -use-dependence-analysis \
///   > output.ll
/// ```
///
///
//===----------------------------------------------------------------------===//
#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include <stdlib.h>
#include <utility>

#include "llvm/Analysis/ValueTracking.h"

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Support/MemoryDependency.h"
#include "llvm/Analysis/ValueTracking.h"

#include "llvm/Analysis/DependenceAnalysis.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "mem-dep-analysis"

using namespace llvm;
using namespace polly;

static cl::opt<bool> useDependenceAnalysis(
    "use-dependence-analysis", cl::init(false),
    cl::desc("Use the memory dependency analysis based on "
             "llvm/Analysis/DependenceAnalysis.h instead of the one based on "
             "polly"));

namespace {

class InstructionDependenceInfo {
public:
  InstructionDependenceInfo(const LoopInfo &li) : loopInfo(li) {}

  /// \brief: Query whether dstInst is has a global in-order instruction
  /// dependence (GIDD) on srcInst, that is, srcInst --GID--> dstInst.
  ///
  /// Reference:
  /// https://ieeexplore.ieee.org/document/8977873
  ///
  /// Returns true if every token coming to dstInst has passed through srcInst
  /// without traversing any BB-edge that would increment common induction
  /// variables
  ///
  /// NOTE: here our goal is to prove the existence of such a dependency (which
  /// can help us eliminating WAR dependency).
  bool hasTokenDependence(Instruction *dstInst, Instruction *srcInst);

  bool hasRevTokenDependence(Instruction *srcInst, Instruction *dstInst);

private:
  const LoopInfo &loopInfo;
};

struct CFGPath {
  std::vector<BasicBlock *> blocks;
  std::map<BasicBlock *, std::set<llvm::Value *>> vals;
};

bool inLoopLatches(const BasicBlock *bb, const std::set<Loop *> &loopSet) {

  return std::any_of(loopSet.begin(), loopSet.end(),
                     [&bb](Loop *loop) { return loop->getLoopLatch() == bb; });
}

/// \brief: Recursive routine that checks if there is always a path from
/// starting from srcInst.
bool instructionAlwaysDepends(const CFGPath &currentPath, Instruction *srcInst,
                              const std::set<Loop *> &loopSet) {
  BasicBlock *curBB = currentPath.blocks.back();
  std::set<llvm::Value *> activeVals = currentPath.vals.at(curBB);
  std::map<BasicBlock *, std::set<llvm::Value *>> phiDepends;

  /// Determine active values of the current basic block
  /// For each instruction in the current basic block, determine whether it
  /// has been marked as an active dependence in a previous call to
  /// tokenDepends (or if path P was initialized with an active dependence).
  /// when an active dependence is found, add all of its own arguments are
  /// themselves added as active dependences.
  for (auto rit = curBB->rbegin(); rit != curBB->rend(); ++rit) {
    auto *inst = &*rit;
    if (isa<BranchInst>(inst) || isa<DbgInfoIntrinsic>(inst))
      continue;

    // If this instruction is an not active dependence, ignore it
    if (activeVals.find(inst) == activeVals.end())
      continue;
    // Else, its operands are active dependences too
    if (auto *phiNode = dyn_cast<PHINode>(inst)) {
      // For Phi nodes, the active dependent values may be different in the
      // different predecessors BB, so we store them in this map for now. We
      // add it to the Path.Val set before recursive calls
      for (auto &predBB : phiNode->blocks()) {
        auto *value = phiNode->getIncomingValueForBlock(predBB);
        if (!(isa<Argument>(value) || isa<Constant>(value)))
          phiDepends[predBB].insert(value);
      }
    } else {
      for (auto *op : inst->operand_values())
        if (!(isa<Argument>(op) || isa<Constant>(op)))
          activeVals.emplace(op);
    }
  }

  bool depends = true;
  if (srcInst->getParent() == curBB) {
    // If through successive tokenDepends calls we have reached the basic
    // block containing I_A, check whether I_A has been added to the list of
    // active dependences. If so, dependency is met.
    depends = activeVals.find(srcInst) != activeVals.end();
  } else {
    SmallVector<BasicBlock *, 2> validPredBBs;
    for (auto *predBB : predecessors(curBB)) {
      // This depends on having a canonical loop structure. Loops will have a
      // single latch with a single successor: the loop header.  Continuing
      // across an edge from a latch to header for any loop in LS is not
      // allowed.
      if (!(inLoopLatches(predBB, loopSet) ||
            (currentPath.vals.count(predBB) &&
             currentPath.vals.at(predBB) == activeVals))) {
        validPredBBs.push_back(predBB);
      }
    }

    // If we have reached a point where active values has been determined,
    // but there are no valid predecessors to produce these values, then
    // dependency is not met.
    depends = !validPredBBs.empty();

    // Else, for each predecessor block, propagate the active values from
    // this basic block (plus any potential active values from phi-nodes
    // with incoming values for the given predecessor block) into a
    // successive call to tokenDepends.
    for (auto *predBB : validPredBBs) {
      if (!depends)
        break;

      CFGPath predBBPath = currentPath;
      predBBPath.blocks.emplace_back(predBB);
      predBBPath.vals[predBB] = activeVals;
      auto it = phiDepends.find(predBB);
      if (it != phiDepends.end())
        for (const auto &val : it->second)
          predBBPath.vals[predBB].insert(val);

      // NOTE: we use "&" because our aim is that the dependency is always
      // there, regardless of the control flow.
      depends &= instructionAlwaysDepends(predBBPath, srcInst, loopSet);
    }
  }

  return depends;
}

static bool instructionAlwaysRevDepends(CFGPath path, Instruction *dstInst,
                                        std::set<Loop *> &ls) {
  int len = path.blocks.size();
  BasicBlock *curBb = path.blocks.back();
  BasicBlock *predBb = (len > 1) ? path.blocks[len - 2] : nullptr;
  auto activeVals = path.vals[curBb];

  /// Determine active values of the current basic block
  /// For each instruction in the current basic block, its operands are
  /// checked to see whether they are present in the current set of active
  /// dependences.
  /// If so, the instruction which has the operand is itself reverse
  /// dependent.
  for (auto &inst : *curBb) {
    if (isa<BranchInst>(&inst) || isa<DbgInfoIntrinsic>(&inst))
      continue;

    std::vector<Value *> operands;
    if (const auto *pi = dyn_cast<PHINode>(&inst)) {
      /* For a PHI node, the only relevant operand is decided by the
       * prev BB */
      if (predBb != nullptr)
        operands.push_back(pi->getIncomingValueForBlock(predBb));
    } else {
      for (auto *op : inst.operand_values())
        operands.push_back(op);
    }
    /* If any of the operands has revdep on LI, this value does too */
    for (auto *op : operands)
      if (activeVals.find(op) != activeVals.end())
        activeVals.insert(&inst);
  }

  bool depends = true;
  if (dstInst->getParent() == curBb) {
    depends = activeVals.find(dstInst) != activeVals.end();
  } else if (inLoopLatches(curBb, ls)) {
    /* This depends on having a canonical loop structure. Loops will
     * have a single latch with a single successor: the loop header.
     * Continuing across an edge from a latch to header for any loop in
     * LS is not allowed. */
  } else {
    const unsigned numSucc = curBb->getTerminator()->getNumSuccessors();
    depends = (numSucc > 0);

    for (auto *succBB : successors(curBb)) {
      if (!depends)
        break;

      /* Skip successor BB if no active values have been added in this
       * call to TokenRevDepends */
      if (std::find(path.blocks.begin(), path.blocks.end(), succBB) !=
          path.blocks.end()) {
        if (path.vals[succBB] == activeVals)
          continue;
      }

      /* If next BB has not been sufficiently explored, explore again */
      CFGPath succBBPath = path;
      succBBPath.blocks.push_back(succBB);
      succBBPath.vals[succBB] = activeVals;

      depends &= instructionAlwaysRevDepends(succBBPath, dstInst, ls);
    }
  }
  return depends;
}

bool InstructionDependenceInfo::hasTokenDependence(Instruction *dstInst,
                                                   Instruction *srcInst) {
  CFGPath cfgPath;
  auto *bb = dstInst->getParent();
  cfgPath.blocks.push_back(bb);
  cfgPath.vals[bb] = {dstInst};
  auto loopSet = std::set<Loop *>();
  for (Loop *loop = loopInfo.getLoopFor(dstInst->getParent()); loop != nullptr;
       loop = loop->getParentLoop())
    loopSet.insert(loop);
  return instructionAlwaysDepends(cfgPath, srcInst, loopSet);
}

bool InstructionDependenceInfo::hasRevTokenDependence(Instruction *srcInst,
                                                      Instruction *dstInst) {
  CFGPath p;
  auto *bb = srcInst->getParent();
  p.blocks.push_back(bb);
  p.vals[bb] = {srcInst};

  auto loopSet = std::set<Loop *>();
  for (Loop *loop = loopInfo.getLoopFor(dstInst->getParent()); loop != nullptr;
       loop = loop->getParentLoop())
    loopSet.insert(loop);

  return instructionAlwaysRevDepends(p, dstInst, loopSet);
}

std::optional<int64_t> getDistance(Dependence *d) {
  unsigned innerMostLoopLevel = d->getLevels();

  if (innerMostLoopLevel == 0) {
    LLVM_DEBUG({
      llvm::errs() << "The two memory accesses are not surrounded by any "
                      "common loops, returning a null distance.\n";
    });
    return std::nullopt;
  }

  const auto *dist = d->getDistance(innerMostLoopLevel);

  LLVM_DEBUG(llvm::errs() << "Distance cannot be calculated at all!");
  if (!dist)
    return std::nullopt;

  if (const auto *distScev = dyn_cast<SCEVConstant>(dist)) {
    auto apInt = distScev->getAPInt();
    // llvm::errs() << "Dist (const): " << apInt.getSExtValue() << "\n";
    return apInt.getSExtValue();
  }

  LLVM_DEBUG(
      llvm::errs() << "Conservatively choose a value based on BB ordering!";);
  LLVM_DEBUG(d->dump(llvm::errs()););
  return std::nullopt;
}

} // namespace

using InstPairType = std::pair<Instruction *, Instruction *>;

namespace {

/// \brief: What a memory analysis has been able to establish about one
/// ordered pair of memory accesses.
///
/// Every pair starts out as `Unknown` and each analysis may refine it into one
/// of the two proven states. A pair that is still `Unknown` once every
/// analysis has run carries no information, and must therefore be treated
/// conservatively (i.e., as if the dependence were real).
enum class DependenceState {
  /// No analysis has been able to say anything about this pair yet.
  Unknown = 0,
  /// The dependence definitely exists: the two accesses may touch the same
  /// address, and their relative order has to be enforced at runtime.
  ProvenTrue,
  /// The dependence definitely does not exist: either the accesses can never
  /// touch the same address, or their order is already enforced by something
  /// else (e.g., by the dataflow itself).
  ProvenFalse,
};

/// \brief: A short, human-readable spelling of a dependence state, for
/// analyses that want to log individual decisions.
[[maybe_unused]] llvm::StringRef toString(DependenceState state) {
  switch (state) {
  case DependenceState::Unknown:
    return "unknown";
  case DependenceState::ProvenTrue:
    return "proven-true";
  case DependenceState::ProvenFalse:
    return "proven-false";
  }
  llvm_unreachable("unhandled DependenceState");
}

/// \brief: A single character standing for a dependence state, used when
/// printing the matrix as a grid.
char toChar(DependenceState state) {
  switch (state) {
  case DependenceState::Unknown:
    return '?';
  case DependenceState::ProvenTrue:
    return 'T';
  case DependenceState::ProvenFalse:
    return 'F';
  }
  llvm_unreachable("unhandled DependenceState");
}

/// \brief: Collects every load and store of a function, in program order.
std::vector<Instruction *> collectMemoryAccesses(Function &llvmFunction) {
  std::vector<Instruction *> accesses;
  for (BasicBlock &bb : llvmFunction)
    for (Instruction &inst : bb)
      if (isa<LoadInst, StoreInst>(inst))
        accesses.push_back(&inst);
  return accesses;
}

/// \brief: The state of every ordered pair of memory accesses in a function.
///
/// The matrix is indexed by (source, destination), where the source is the
/// *predecessor* of the dependence and the destination its *successor*: the
/// entry at `(src, dst)` answers "must `src` be ordered before `dst`?". It is
/// therefore asymmetric, and `(src, dst)` and `(dst, src)` are two independent
/// entries; the diagonal is present but meaningless, since an access is never
/// a dependence of itself.
///
/// This is meant to be the shared substrate for the various analyses in this
/// pass: each analysis refines the entries it can prove, and the final set of
/// dependence edges is read off the matrix once they have all run.
class DependenceMatrix {
public:
  /// Builds a matrix covering `accesses`, with every entry set to `Unknown`.
  /// Accesses are indexed in the order in which they are given.
  explicit DependenceMatrix(llvm::ArrayRef<Instruction *> accesses)
      : accesses(accesses.begin(), accesses.end()),
        states(accesses.size() * accesses.size(), DependenceState::Unknown) {
    for (auto [idx, access] : llvm::enumerate(this->accesses))
      accessToIndex.try_emplace(access, idx);
    // Every entry starts out `Unknown`, so every entry starts out in the set.
    for (unsigned flatIndex = 0; flatIndex < states.size(); ++flatIndex)
      unknownEntries.insert(unknownEntries.end(), flatIndex);
  }

  /// The number of memory accesses covered, i.e., the side length of the
  /// matrix.
  unsigned getNumAccesses() const { return accesses.size(); }

  /// The memory accesses covered, in index order.
  llvm::ArrayRef<Instruction *> getAccesses() const { return accesses; }

  /// The index of an access, which must be one the matrix covers.
  unsigned getIndexOf(Instruction *access) const {
    auto it = accessToIndex.find(access);
    assert(it != accessToIndex.end() &&
           "memory access is not covered by this dependence matrix");
    return it->second;
  }

  /// \brief: What is currently known about the dependence srcAccess ->
  /// dstAccess.
  DependenceState getState(Instruction *srcAccess,
                           Instruction *dstAccess) const {
    return states[getFlatIndex(getIndexOf(srcAccess), getIndexOf(dstAccess))];
  }
  DependenceState getState(unsigned srcIndex, unsigned dstIndex) const {
    return states[getFlatIndex(srcIndex, dstIndex)];
  }

  /// \brief: Records what an analysis has established about the dependence
  /// srcAccess -> dstAccess. Keeps `unknownEntries` in step.
  void setState(unsigned srcIndex, unsigned dstIndex, DependenceState state) {
    unsigned flatIndex = getFlatIndex(srcIndex, dstIndex);
    if (state == DependenceState::Unknown)
      unknownEntries.insert(flatIndex);
    else
      unknownEntries.erase(flatIndex);
    states[flatIndex] = state;
  }
  void setState(Instruction *srcAccess, Instruction *dstAccess,
                DependenceState state) {
    setState(getIndexOf(srcAccess), getIndexOf(dstAccess), state);
  }

  /// The number of entries no analysis has settled yet.
  unsigned getNumUnknown() const { return unknownEntries.size(); }

  /// \brief: The dependences that are still `Unknown`, as (source,
  /// destination) access pairs, in the same row-major order as the matrix.
  ///
  /// This is what a refinement pass should loop over: it costs the number of
  /// entries still open rather than a full rescan of the matrix, and entries
  /// an earlier pass has already settled are skipped for free.
  ///
  /// The result is a snapshot, so refining entries while looping over it is
  /// safe. Reading the live set instead would not be: settling an entry
  /// erases it from `unknownEntries`, which invalidates an iterator to it.
  std::vector<InstPairType> getUnknownDependences() const {
    std::vector<InstPairType> unknownPairs;
    unknownPairs.reserve(unknownEntries.size());

    for (unsigned flatIndex : unknownEntries)
      unknownPairs.emplace_back(accesses[flatIndex / getNumAccesses()],
                                accesses[flatIndex % getNumAccesses()]);

    return unknownPairs;
  }

  /// \brief: The dependence edges the matrix currently describes.
  ///
  /// An edge is produced for every ordered pair that could be a dependence and
  /// has not been proven not to be one, so a pair that is still `Unknown`
  /// conservatively becomes an edge.
  ///
  /// This reads nothing but the matrix: every reason to drop a pair has to
  /// have been written into it as `ProvenFalse` by one of the refinement
  /// passes beforehand (see removeEqual, removeRAR, removeUnequalBase).
  std::vector<InstPairType> getDependencePairs() const {
    std::vector<InstPairType> depPairList;

    for (Instruction *srcAccess : accesses)
      for (Instruction *dstAccess : accesses)
        if (getState(srcAccess, dstAccess) != DependenceState::ProvenFalse)
          depPairList.emplace_back(srcAccess, dstAccess);

    return depPairList;
  }

  /// \brief: The same edges, grouped by source access and keyed the way the
  /// serialization to LLVM metadata expects.
  ///
  /// One memory access can be the source of several dependences. E.g.,
  ///
  ///   store %location, %data; name = "store1"
  ///   %read_data1 = load %location; name = "load1"
  ///   %read_data2 = load %location; name = "load2"
  ///
  /// has two RAW dependences, (store1, load1) and (store1, load2), and both
  /// are annotated on store1 as a single list [dep1, dep2]. Each source
  /// therefore maps to one `LLVMMemDependency` holding all of its
  /// destinations.
  ///
  /// `nameMapping` must name every access the matrix covers, as produced by
  /// `nameAllLoadStores`.
  std::map<Instruction *, LLVMMemDependency> toDependencyMap(
      const std::map<Instruction *, std::string> &nameMapping) const;

  /// \brief: Prints the matrix as a grid of one character per entry ('T' for
  /// proven true, 'F' for proven false, '?' for unknown), rows being sources
  /// and columns destinations, preceded by the legend of access indices.
  void print(llvm::raw_ostream &os) const {
    os << "Dependence matrix over " << getNumAccesses() << " memory accesses, "
       << getNumUnknown()
       << " entries still unknown (rows: source/predecessor, columns: "
          "destination/successor):\n";
    for (auto [idx, access] : llvm::enumerate(accesses))
      os << "  [" << idx << "]" << *access << "\n";
    for (unsigned srcIndex = 0; srcIndex < getNumAccesses(); ++srcIndex) {
      os << "  ";
      for (unsigned dstIndex = 0; dstIndex < getNumAccesses(); ++dstIndex)
        os << toChar(getState(srcIndex, dstIndex));
      os << "\n";
    }
  }

private:
  /// The offset of an entry in the row-major `states` array.
  unsigned getFlatIndex(unsigned srcIndex, unsigned dstIndex) const {
    assert(srcIndex < getNumAccesses() && dstIndex < getNumAccesses() &&
           "dependence matrix index out of range");
    return srcIndex * getNumAccesses() + dstIndex;
  }

  /// The accesses covered, in index order.
  llvm::SmallVector<Instruction *> accesses;
  /// The reverse of `accesses`.
  llvm::DenseMap<Instruction *, unsigned> accessToIndex;
  /// The entries, stored row-major: the state of (srcIndex, dstIndex) lives at
  /// `srcIndex * getNumAccesses() + dstIndex`.
  std::vector<DependenceState> states;
  /// The flat indices of the entries that are still `Unknown`, maintained by
  /// setState() so that a refinement pass never has to rescan the matrix to
  /// find the work it has left. Ordered, so iteration is deterministic and
  /// follows the same row-major order as `states`.
  std::set<unsigned> unknownEntries;
};

} // namespace

/// \brief: can this dependency be violated in the same iteration of a
/// loop/outside of a loop? Either:
/// - LLVM aliasanalysis finds that there is no alias, so they cannot touch the
///  same address, or
/// - the successor is semantically before the predecessor: In this case there
/// is no conflict if the successor runs before the predecessor
bool canBeViolatedInSameIteration(Instruction *srcAccess,
                                  Instruction *dstAccess,
                                  AAManager::Result &aliasAnalysis,
                                  const LoopInfo &loopInfo) {
  // Can they touch the same address at all, for one valuation of the
  // surrounding induction variables?
  if (aliasAnalysis.alias(MemoryLocation::get(srcAccess),
                          MemoryLocation::get(dstAccess)) ==
      AliasResult::NoAlias)
    return false;

  // Inside one basic block, execution order within an iteration is exactly
  // program order.
  if (srcAccess->getParent() == dstAccess->getParent())
    return srcAccess->comesBefore(dstAccess);

  // Across blocks, a path that stays inside one iteration is one that never
  // goes around a back edge of a loop the two accesses share, so blocking
  // those latches turns plain CFG reachability into the same-iteration
  // question. When they share no loop, the latch set is empty and this
  // degenerates to "can dstAccess ever follow srcAccess".
  llvm::SmallPtrSet<BasicBlock *, 4> backEdgeSources;
  for (const Loop *loop = loopInfo.getLoopFor(srcAccess->getParent());
       loop != nullptr; loop = loop->getParentLoop()) {
    if (!loop->contains(dstAccess->getParent()))
      continue;
    llvm::SmallVector<BasicBlock *, 4> latches;
    loop->getLoopLatches(latches);
    backEdgeSources.insert(latches.begin(), latches.end());
  }

  return isPotentiallyReachable(srcAccess, dstAccess, &backEdgeSources);
}

/// \brief: An data container class that represents the analysis data from the
/// Scop.
/// https://www.cs.colostate.edu/~pouchet/software/polyopt/doc/htmltexinfo/Specifics-of-Polyhedral-Programs.html.
class ScopAnalysisInfo {
  LoopInfo *loopInfo;

  int scopMinDepth;
  std::vector<Instruction *> memInsts;
  std::map<Instruction *, isl::map> instToCurrentMap;
  std::map<Instruction *, int> instToLoopDepth;
  std::map<Instruction *, llvm::Value *> instToBase;
  /// Each Minimized Scop has a separate context. This ensures that trying to
  /// intersect maps for instructions from separate Scops will raise an error
  isl::ctx ctx;
  // Used by the dependsInternal() function
  std::map<InstPairType, bool> dependsCache;
  std::set<InstPairType> outstandingDependsQueries;

  /// \brief (needs proof-read here): Find the loop depth of the inner most
  /// common loop that contains both instructions.
  int getInnerMostCommonLoopDepth(Instruction *i0, Instruction *i1) {
    const auto *bb0 = i0->getParent();
    const auto *bb1 = i1->getParent();
    int depth0 = loopInfo->getLoopDepth(bb0);
    int depth1 = loopInfo->getLoopDepth(bb1);
    Loop *l0 = loopInfo->getLoopFor(bb0);
    Loop *l1 = loopInfo->getLoopFor(bb1);

    // NOTE: These two while loops attempt to find the common loop (not
    // necessarily the outer-most) that contains both instructions.
    while (depth0 > depth1) {
      l0 = l0->getParentLoop();
      depth0--;
    }
    while (depth1 > depth0) {
      l1 = l1->getParentLoop();
      depth1--;
    }

    // NOTE: Keep reducing loop depths until they match, or we reach outside all
    // loops (i.e., depth0 == 0).
    while (l1 != l0 && depth0-- > 0) {
      l0 = l0->getParentLoop();
      l1 = l1->getParentLoop();
    }

    return depth0;
  }

  /// \brief: Is this access one of the ones this Scop covers?
  bool isCovered(Instruction *inst) const {
    return instToCurrentMap.count(inst) > 0;
  }

  isl::map getMap(Instruction *inst, unsigned int depthToKeep, bool getFuture) {

    const auto currentMap = instToCurrentMap[inst];

    auto inDimsToBeChecked = currentMap.dim(isl::dim::in);

    if (inDimsToBeChecked.is_error())
      llvm::report_fatal_error(
          "Fail to extraction the input dim of currentMap!");

    unsigned inDimValue = static_cast<unsigned>(inDimsToBeChecked);

    assert(inDimValue >= depthToKeep);

    isl::map retMap = currentMap.project_out(isl::dim::in, depthToKeep,
                                             inDimValue - depthToKeep);
    if (getFuture && depthToKeep > 0) {
      retMap = makeFutureMap(retMap);
    }
    return removeMapMeta(retMap);
  }

  /// \brief: Functions for modifying isl::map to future forms
  isl::map makeFutureMap(const isl::map &map) {
    isl::map fMap, tmpMap;

    auto nInsToBeChecked = map.dim(isl::dim::in);

    if (nInsToBeChecked.is_error())
      llvm::report_fatal_error("Cannot extract the input dim of map!");

    unsigned nIns = static_cast<unsigned>(nInsToBeChecked);

    /* Add input vars */
    tmpMap = map.add_dims(isl::dim::in, nIns);
    /* Add future constraints on new input variables */
    for (unsigned int i = 1; i <= nIns; i++) {
      isl::map constrMapN = addFutureCondition(tmpMap, i);
      if (i == 1)
        fMap = constrMapN;
      else
        fMap = fMap.unite(constrMapN);
    }
    /* Project out old input vars */
    fMap = fMap.project_out(isl::dim::in, 0, nIns);
    assert(fMap.get() != nullptr);
    return fMap;
  }

  /// \brief: Add constraints on the 'n' most significant dimensions
  isl::map addFutureCondition(const isl::map &map, int n) {

    auto nInsToBeChecked = map.dim(isl::dim::in);

    if (nInsToBeChecked.is_error())
      llvm::report_fatal_error("Cannot extract input dimension of map!");

    int nIns = static_cast<unsigned>(nInsToBeChecked) / 2;
    isl::map constrMap = map;

    isl_local_space *lsp =
        isl_local_space_from_space(map.get_space().release());
    isl::local_space ls = isl::manage(lsp);

    // Add equality constraints on the first 'n - 1' dims, Inequality on the
    // last dim
    for (int i = 0; i < n; i++) {
      isl::constraint c;
      if (i == n - 1) {
        c = isl::constraint::alloc_inequality(ls);
        c = c.set_constant_si(-1);
      } else
        c = isl::constraint::alloc_equality(ls);
      c = c.set_coefficient_si(isl::dim::in, i, 1);
      c = c.set_coefficient_si(isl::dim::in, i + nIns, -1);
      constrMap = constrMap.add_constraint(c);
    }

    return isl::map(constrMap);
  }

  isl::map copyMapMeta(isl::map map, const isl::map &templateMap) {

    isl::id inTupleID = templateMap.get_tuple_id(isl::dim::in);
    isl::id outTupleID = templateMap.get_tuple_id(isl::dim::out);

    map = map.set_tuple_id(isl::dim::in, inTupleID);
    map = map.set_tuple_id(isl::dim::out, outTupleID);

    return map;
  }

  isl::map removeMapMeta(isl::map map) {
    auto emptyID = isl::id::alloc(ctx, "", nullptr);

    map = map.set_tuple_id(isl::dim::in, emptyID);
    map = map.set_tuple_id(isl::dim::out, emptyID);

    return map;
  }

public:
  ScopAnalysisInfo(Scop &scop) : ctx(isl::ctx(isl_ctx_alloc())) {
    loopInfo = scop.getLI();

    // @Jiahui17: Here is my understanding of what the code below is doing, we
    // need a person to proof-read this.
    //
    // clang-format off
    // Calculate scopMinDepth based on first scopStmt.
    // example:
    // for (...) { // <- This is the start of the full loop nest (getRelativeLoopDepth will factor this part out)
    //   if (A[0] > 1) {
    //     // Scop starts from here: notice that, here, by definition, the depth is 1 (hence the assert below)
    //     for (i=0;i<N;++i) { 
    //       tmp_A = A[i][0]; // <- first scop statement (the code below calculates the depth of this??)
    //       for (j=0;j<M;++j) {
    //         tmp_B = B[i][j]; // <- second scop statement
    //         tmp_C = C[i][j]; 
    //         tmp = tmp_A + tmp_B + tmp_C;
    //         D[i][j] = tmp_A;
    //         ...
    //       }
    //     }
    //     // Scop ends at here
    //   }
    // }
    // clang-format on

    auto *bb = scop.begin()->getBasicBlock();
    auto *loop = loopInfo->getLoopFor(bb);
    scopMinDepth = loopInfo->getLoopDepth(bb) - scop.getRelativeLoopDepth(loop);
    assert(scopMinDepth > 0);
  }

  ~ScopAnalysisInfo() = default;

  /// \brief: Use addScopStmt() to add all ScopStmt's in a Scop, then
  /// refineDependences() to write what this Scop proves into the matrix.
  void addScopStmt(ScopStmt &stmt) {
    int depth = loopInfo->getLoopDepth(stmt.getBasicBlock());

    for (auto *inst : stmt.getInstructions())
      // NOTE (@Jiahui17): the call `memcpy` (or any other function that may
      // access the memory more than once) in the main function might be
      // analyzed here and trigger an assertion error. Therefore, CallInst is
      // ignored in our memory dependency analysis
      if (inst->mayReadOrWriteMemory() && !isa<CallInst>(inst)) {
        auto &memoryAccess = stmt.getArrayAccessFor(inst);

        isl::map currentMap = memoryAccess.getLatestAccessRelation();

        isl::map domain = isl::map::from_domain(stmt.getDomain());

        auto outDim = currentMap.dim(isl::dim::out);

        if (outDim.is_error())
          llvm::report_fatal_error("Failed to extract output dimension");

        domain = domain.add_dims(isl::dim::out, static_cast<unsigned>(outDim));

        domain = copyMapMeta(domain, currentMap);

        instToCurrentMap.emplace(inst, currentMap.intersect(domain));

        instToLoopDepth[inst] = depth;
        instToBase[inst] = memoryAccess.getOriginalBaseAddr();
        memInsts.push_back(inst);
      }
  }

  // clang-format off

// \brief: Refines `depMatrix` with the polyhedral information of this Scop.
//
// For each pair of accesses that is still unknown, it intersects the sets of
// addresses the two of them touch. An empty intersection proves they can
// never conflict, so the dependence is `ProvenFalse`. A non-empty one proves
// nothing (the accesses may collide)
//
// Example:
// 1. For ... -> SI -> LI -> ... , SI may affect LI in this and future iterations
//          ↱---------------↵
// intersect store-set with current and future load-set.
//
// 2. For ... -> LI -> SI -> ... , SI may affect LI only in future iterations
//           ↱---------------↵
// intersect store-set with future load-set.
//
// 3. For ... -> SI -> ......     , SI and LI iterations are independent.
//         ↱  -> LI ->     |
//         |---------------↵
// intersect entire store-set with entire load-set.

  // clang-format on
  void refineDependences(DependenceMatrix &depMatrix,
                         AAManager::Result &aliasAnalysis) {
    for (auto [srcAccess, dstAccess] : depMatrix.getUnknownDependences()) {
      // This Scop can only speak about the accesses it covers. Anything else
      // stays unknown, and some other analysis (or the conservative default)
      // has to deal with it.
      if (!isCovered(srcAccess) || !isCovered(dstAccess))
        continue;

      isl::map srcMap, dstMap;

      // If the dependence can be violated within one iteration, that
      // iteration has to stay in the intersection
      if (canBeViolatedInSameIteration(srcAccess, dstAccess, aliasAnalysis,
                                       *loopInfo)) {
        // We cannot put any restrictions on the indices being processed by
        // the instructions, so we intersect the sets of all possible indices
        // ever accessed.
        srcMap = getMap(srcAccess, 0, false);
        dstMap = getMap(dstAccess, 0, false);
      } else {
        int commonDepth = getInnerMostCommonLoopDepth(srcAccess, dstAccess);

        // The two share no loop, so there are no other iterations for them
        // to collide in, and alias analysis has just ruled out the only
        // occasion they both execute. NOTE: this leans on the same
        // cross-loop assumption as removeNonAliasing.
        if (commonDepth == 0 && scopMinDepth == 1) {
          depMatrix.setState(srcAccess, dstAccess,
                             DependenceState::ProvenFalse);
          continue;
        }

        // Only a later iteration of the destination can conflict, so the
        // destination's set is taken over the future iterations only.
        assert(commonDepth - scopMinDepth + 1 >= 0);
        unsigned depthToKeep = commonDepth - scopMinDepth + 1;
        srcMap = getMap(srcAccess, depthToKeep, false);
        dstMap = getMap(dstAccess, depthToKeep, true);
      }

      // Only an answer of "definitely empty" disproves the dependence; an isl
      // error leaves the entry unknown.
      if (srcMap.intersect(dstMap).is_empty().is_true())
        depMatrix.setState(srcAccess, dstAccess, DependenceState::ProvenFalse);
    }
  }

  std::map<Instruction *, llvm::Value *> &getInstsToBase() {
    return instToBase;
  }

  using iterator = std::vector<Instruction *>::iterator;
  iterator begin() { return memInsts.begin(); }
  iterator end() { return memInsts.end(); }
};

struct IndexAnalysis {

  IndexAnalysis() : otherInsts() {}
  ~IndexAnalysis() = default;

  /// Returns all memory instructions in SCoPs which do not require an LSQ
  /// connection
  std::vector<Instruction *> &getOtherInsts() { return otherInsts; }

  /// Query whether any SCoP contains BB
  bool isInScop(BasicBlock *bb) { return bbList.find(bb) != bbList.end(); }

  /// Returns an integer uniquely identifying the SCoP which contains BB
  int getScopID(BasicBlock *bb) {
    return (isInScop(bb)) ? bbToScopMap[bb] : -1;
  }

  std::vector<Instruction *> otherInsts;

  // NOTE: in the legacy implementation they were called "instRAWlist". But this
  // was actually imprecise, as this contains also RAW dependencies.
  std::set<InstPairType> dependentReadAndWritePairs;
  std::set<InstPairType> dependentWriteAndWritePairs;
  std::set<BasicBlock *> bbList;
  std::map<BasicBlock *, int> bbToScopMap;
  std::map<Instruction *, Value *> instToBase;
};

namespace {

void getAllRegions(llvm::Region &region,
                   std::deque<llvm::Region *> &regionQueue) {
  regionQueue.push_back(&region);
  for (const auto &e : region)
    getAllRegions(*e, regionQueue);
}

bool hasMemoryReadOrWrite(ScopStmt &stmt) {
  bool hasRdWr = false;
  for (auto *inst : stmt.getInstructions()) {
    hasRdWr |= inst->mayReadOrWriteMemory();
  }
  return hasRdWr;
}

// Returns the base address produced by the alloca instruction or the global
// constant declaration.
const Value *findBaseInternal(Value *addr) {
  llvm::SmallVector<const llvm::Value *, 2> baseArray;
  getUnderlyingObjects(addr, baseArray);
  if (baseArray.empty()) {
    llvm::report_fatal_error(
        "Cannot determine the base array of the load operation! Aborting...");
  } else if (baseArray.size() > 1) {
    LLVM_DEBUG({
      llvm::errs()
          << "The index value is calculated from multiple base addresses!\n";
      llvm::errs() << "List of addresses:\n";
      for (const auto *addr : baseArray) {
        addr->dump();
      }
    });
    llvm::report_fatal_error(
        "The index value is calculated from multiple distinct base "
        "addresses. This is a currently unsupported IR construction.");
  }
  return baseArray[0];
}

const Value *findBase(Instruction *inst) {
  Value *addr;
  if (auto *loadInst = dyn_cast<LoadInst>(inst)) {
    addr = loadInst->getPointerOperand();
  } else if (auto *storeInst = dyn_cast<StoreInst>(inst)) {
    addr = storeInst->getPointerOperand();
  } else {
    llvm_unreachable("Instruction is not a memory access");
  }

  return findBaseInternal(addr);
}

bool equalBase(Instruction *a, Instruction *b) {
  return findBase(a) == findBase(b);
}

/// \brief: Rules out every pair of an access with itself.
///
/// An access is never a dependence of itself: the diagonal of the matrix is
/// meaningless, and the rest of the pipeline treats a self-edge as a WAW
/// between two executions of one instruction, which the dataflow already
/// orders (see MarkMemoryInterfaces).
void removeEqual(DependenceMatrix &depMatrix) {
  for (Instruction *access : depMatrix.getAccesses())
    depMatrix.setState(access, access, DependenceState::ProvenFalse);
}

/// \brief: Rules out every read-after-read pair.
///
/// Two loads never conflict, and the rest of the pipeline assumes RAR edges
/// are never recorded.
void removeRAR(DependenceMatrix &depMatrix) {
  for (auto [srcAccess, dstAccess] : depMatrix.getUnknownDependences())
    if (!srcAccess->mayWriteToMemory() && !dstAccess->mayWriteToMemory())
      depMatrix.setState(srcAccess, dstAccess, DependenceState::ProvenFalse);
}

/// \brief: Rules out every pair of accesses to different base arrays.
///
/// Dynamatic places distinct base arrays in distinct RAMs, so two accesses to
/// different arrays can never conflict. They also end up on different memory
/// interfaces, and an edge across two interfaces would trip the assertions in
/// MemoryInterfaces.
///
/// NOTE: Needs to be ran after removeRAR and removeEqual as
/// equalBase can throw an error for same access and two load queries
void removeUnequalBase(DependenceMatrix &depMatrix) {
  for (auto [srcAccess, dstAccess] : depMatrix.getUnknownDependences())
    if (!equalBase(srcAccess, dstAccess))
      depMatrix.setState(srcAccess, dstAccess, DependenceState::ProvenFalse);
}

/// \brief: Do the two accesses sit inside at least one common loop?
bool haveCommonLoop(const LoopInfo &loopInfo, Instruction *a, Instruction *b) {
  std::set<const Loop *> loopsOfA;
  for (const Loop *loop = loopInfo.getLoopFor(a->getParent()); loop != nullptr;
       loop = loop->getParentLoop())
    loopsOfA.insert(loop);

  for (const Loop *loop = loopInfo.getLoopFor(b->getParent()); loop != nullptr;
       loop = loop->getParentLoop())
    if (loopsOfA.count(loop) > 0)
      return true;

  return false;
}

/// \brief: Rules out the pairs whose successor can never execute after their
/// predecessor at all.
///
/// This is the "not in a loop" case of the rule that a dependence whose
/// predecessor runs after its successor is vacuous: in
///
///   A[j] = ...;
///   A[k] = ...;
///
/// only A[j] -> A[k] has to be enforced. For two accesses inside one loop
/// this never fires, because the back edge makes each reachable from the
/// other; there the same rule is applied per iteration, inside
/// canBeViolatedInSameIteration.
void removeSucceedingPredecessor(DependenceMatrix &depMatrix) {
  for (auto [srcAccess, dstAccess] : depMatrix.getUnknownDependences())
    if (!isPotentiallyReachable(srcAccess, dstAccess))
      depMatrix.setState(srcAccess, dstAccess, DependenceState::ProvenFalse);
}

/// \brief: Rules out the pairs that cannot be violated within one iteration
/// and have no other iteration in which to be violated.
void removeNonAliasing(DependenceMatrix &depMatrix,
                       AAManager::Result &aliasAnalysis,
                       const LoopInfo &loopInfo) {
  for (auto [srcAccess, dstAccess] : depMatrix.getUnknownDependences()) {
    if (haveCommonLoop(loopInfo, srcAccess, dstAccess))
      continue;

    if (!canBeViolatedInSameIteration(srcAccess, dstAccess, aliasAnalysis,
                                      loopInfo))
      depMatrix.setState(srcAccess, dstAccess, DependenceState::ProvenFalse);
  }
}

std::map<Instruction *, LLVMMemDependency> DependenceMatrix::toDependencyMap(
    const std::map<Instruction *, std::string> &nameMapping) const {
  std::map<Instruction *, LLVMMemDependency> instToDepsMap;

  for (auto &[srcAccess, dstAccess] : getDependencePairs()) {
    assert(nameMapping.count(srcAccess) > 0 && "Unnamed load/store op!");
    assert(nameMapping.count(dstAccess) > 0 && "Unnamed load/store op!");

    LLVMMemDependency &deps = instToDepsMap[srcAccess];
    deps.name = nameMapping.at(srcAccess);
    deps.destAndDepthAndDist.emplace_back(nameMapping.at(dstAccess), 0, 0);
  }

  return instToDepsMap;
}

/// \brief: an LLVM pass that combines polyhedral and alias analysis to compute
/// a set of dependency edges from the LLVM IR. It further uses dataflow
/// analysis to eliminate dependency edges enforced by the dataflow.
struct MemDepAnalysisPass : PassInfoMixin<MemDepAnalysisPass> {

  // This struct keeps track for every memory instruction:
  // - Is it in any scop (i.e., it is in `instToScopId`)?
  // - Are two instructions in the same scop?
  struct SameScopHelper {
    std::map<Instruction *, int> instToScopId;
    bool sameScop(Instruction *a, Instruction *b) const {
      if (instToScopId.count(a) == 0)
        return false;
      if (instToScopId.count(b) == 0)
        return false;

      return (instToScopId.at(a) == instToScopId.at(b));
    }
  };

  IndexAnalysis indexAnalysis;
  AAManager::Result *aliasAnalysis;
  unsigned memCount = 0;

  /// \brief: Applies the polyhedral and dataflow analysis of one Scop to the
  /// dependence matrix, disproving the pairs it can.
  void processScop(Scop &scop, DependenceMatrix &depMatrix,
                   AAManager::Result &aliasAnalysis);

  /// \brief: Loops through the loops in the IR and collect the loads and
  /// stores.
  void processLoop(Loop *l, std::vector<struct LoopMetaData> &loopMetaInfos);

  // Uses "llvm/Analysis/DependenceAnalysis.h"
  PreservedAnalyses runDependenceAnalysisBased(Function &llvmFunction,
                                               FunctionAnalysisManager &fam);

  // Polly-based refinement inherited from the legacy Dynamatic
  void refineWithPollyAnalysis(Function &llvmFunction,
                               FunctionAnalysisManager &fam,
                               DependenceMatrix &depMatrix);

  PreservedAnalyses run(Function &llvmFunction, FunctionAnalysisManager &fam);

  /// \brief: returns a list of (srcInst, dstInst) pairs that might have a WAR
  /// or WAW conflict.
  std::vector<InstPairType>
  getDependencyPairs(Function &llvmFunction,
                     const SameScopHelper &sameScopHelper);
  std::map<Instruction *, std::string> nameAllLoadStores(Function &f);
};

std::map<Instruction *, std::string>
MemDepAnalysisPass::nameAllLoadStores(Function &f) {
  llvm::LLVMContext &context = f.getContext();

  std::map<Instruction *, std::string> nameMapping;

  for (llvm::BasicBlock &bb : f) {
    for (llvm::Instruction &instr : bb) {
      if (llvm::LoadInst *loadInstr = llvm::dyn_cast<llvm::LoadInst>(&instr)) {

        std::string name = "load" + std::to_string(memCount);

        // Create a metadata string
        llvm::MDString *mdStr = llvm::MDString::get(context, name);

        // Create an MDNode containing the MDString
        // MDNode::get takes a context and an arrayref of llvm::Value*
        llvm::MDNode *md = llvm::MDNode::get(context, mdStr);

        loadInstr->setMetadata(dynamatic::NameAnalysis::ATTR_NAME, md);
        nameMapping[&instr] = name;
        memCount++;
      } else if (llvm::StoreInst *storeInstr =
                     llvm::dyn_cast<llvm::StoreInst>(&instr)) {

        std::string name = "store" + std::to_string(memCount);

        // Create a metadata string
        llvm::MDString *mdStr = llvm::MDString::get(context, name);

        // Create an MDNode containing the MDString
        llvm::MDNode *md = llvm::MDNode::get(context, mdStr);

        storeInstr->setMetadata(dynamatic::NameAnalysis::ATTR_NAME, md);
        nameMapping[&instr] = name;
        memCount++;
      }
    }
  }
  return nameMapping;
}

void MemDepAnalysisPass::processScop(Scop &scop, DependenceMatrix &depMatrix,
                                     AAManager::Result &aliasAnalysis) {

  ScopAnalysisInfo meta(scop);

  for (auto &stmt : scop) {
    if (!hasMemoryReadOrWrite(stmt))
      continue;

    meta.addScopStmt(stmt);
  }

  meta.refineDependences(depMatrix, aliasAnalysis);
}

// Helper function: Get all instructions of a certain type "T"
template <typename T>
std::vector<Instruction *> getAllInsts(Function *llvmFunction) {
  std::vector<Instruction *> insts;
  for (BasicBlock &bb : *llvmFunction) {
    for (Instruction &inst : bb)
      if (isa<T>(inst))
        insts.push_back(&inst);
  }
  return insts;
}

std::vector<InstPairType>
MemDepAnalysisPass::getDependencyPairs(Function &llvmFunction,
                                       const SameScopHelper &sameScopHelper) {
  std::vector<InstPairType> depPairList;
  for (auto *storeInst : getAllInsts<StoreInst>(&llvmFunction)) {
    // Find RAW dependencies
    for (auto *loadInst : getAllInsts<LoadInst>(&llvmFunction)) {

      InstPairType rawPair = std::make_pair(storeInst, loadInst);

      // NOTE: In dynamatic we assume that memory with different base addresses
      // are store in separate RAMs. Two instructions targetting differing base
      // arrays can never conflict.
      if (!equalBase(storeInst, loadInst))
        continue;

      // Instructions are in the same scop: use the result from IndexAnalysis
      if (sameScopHelper.sameScop(loadInst, storeInst)) {
        if (indexAnalysis.dependentReadAndWritePairs.count(rawPair))
          depPairList.push_back(rawPair);

        LLVM_DEBUG({
          if (!indexAnalysis.dependentReadAndWritePairs.count(rawPair)) {
            llvm::dbgs() << "--------------------------------------------\n";
            llvm::dbgs() << "The following memory access instruction pair "
                            "proven to be independent according to polyhedral "
                            "analysis:\n";
            loadInst->dump();
            storeInst->dump();
          }
        });
        continue;
      }

      // Instruction are in different Scops: use the result from alias analysis
      AliasResult aliasResult = aliasAnalysis->alias(
          MemoryLocation::get(loadInst), MemoryLocation::get(storeInst));

      // If they always or sometimes alias:
      if (aliasResult != AliasResult::NoAlias) {
        // If the pair of load/store potentially access the same memory
        // location, then we consider two cases:
        //   1. If it is possible to reach from the load inst to the store, then
        //   we add the WAR dependency
        //   2. If it is possible to reach from the store inst to the load, then
        //   we add the RAW dep
        if (isPotentiallyReachable(storeInst, loadInst))
          depPairList.emplace_back(storeInst, loadInst);
        if (isPotentiallyReachable(loadInst, storeInst))
          depPairList.emplace_back(loadInst, storeInst);
      }
    }
    // Find WAW dependencies
    for (auto *secondStoreInst : getAllInsts<StoreInst>(&llvmFunction)) {
      if (secondStoreInst == storeInst)
        continue;

      // NOTE: In dynamatic we assume that memory with different base addresses
      // are store in separate RAMs. Two instructions targetting differing base
      // arrays can never conflict.
      if (!equalBase(storeInst, secondStoreInst))
        continue;

      auto pair = InstPairType(secondStoreInst, storeInst);
      auto pairRev = InstPairType(storeInst, secondStoreInst);

      // Instructions are in the same scop: use the result from IndexAnalysis
      if (sameScopHelper.sameScop(storeInst, secondStoreInst)) {
        if (indexAnalysis.dependentWriteAndWritePairs.count(pair) > 0)
          depPairList.push_back(pair);
        else if (indexAnalysis.dependentWriteAndWritePairs.count(pairRev) > 0)
          depPairList.push_back(pairRev);
        continue;
      }

      // Otherwise, use results from alias analysis:
      AliasResult aliasResult = aliasAnalysis->alias(
          MemoryLocation::get(storeInst), MemoryLocation::get(secondStoreInst));
      // If they always or sometimes alias:
      if (aliasResult != AliasResult::NoAlias) {
        depPairList.push_back(pair);
      }
    }
  }

  return depPairList;
}

// This flow currently only supports the dependence analysis within one BB and
// the inner-most loops in all loop nests
PreservedAnalyses
MemDepAnalysisPass::runDependenceAnalysisBased(Function &llvmFunction,
                                               FunctionAnalysisManager &fam) {

  // REMARK:
  //
  // We aim to use this analysis to report:
  // (predecessor, successor, iteration-dist)
  //
  // which means that predecessor in iteration i has to go before successor in
  // iteration i + "iteration-dist"
  //
  // Example:
  // (ld, st, 1)
  // this means that the ld in iteration i has to go before st in iteration i +
  // 1
  //
  // So, we need to rely on their BB order to understand who actually goes
  // first.
  auto &dependenceAnalysis = fam.getResult<DependenceAnalysis>(llvmFunction);
  auto nameMapping = nameAllLoadStores(llvmFunction);

  if (llvmFunction.getName() == "main") {
    return PreservedAnalyses::all();
  }

  std::vector<Instruction *> memoryInsts;
  for (BasicBlock &bb : llvmFunction) {
    for (Instruction &inst : bb)
      if (isa<LoadInst, StoreInst>(inst))
        memoryInsts.push_back(&inst);
  }

  // This map is used to record the dependencies and we serialize them to
  // metadata nodes later.
  std::map<Instruction *, LLVMMemDependency> instToDepsMap;

  // REMARK:
  // - When the dependency analysis reports a loop independent dependency, it
  // actually tests positive for both RAW and WAR; in this case,  we use the BB
  // order to determine which one goes first.
  // - When the dependency analysis reports a loop-carried dependency, it will
  // correctly reports both direction with one positive and one negative.
  //
  for (auto *src : memoryInsts) {
    for (auto *dst : memoryInsts) {
      if (src->getParent() != dst->getParent())
        continue;

      // Check the base address pointer used in gep of the two accesses (e.g.,
      // function arguments, allocas..), we assume that different function
      // arguments do not alias.
      if (!equalBase(src, dst))
        continue;

      if (src == dst)
        continue;

      if (auto d = dependenceAnalysis.depends(/* proceed */ src,
                                              /* succeed */ dst, true)) {

        auto llvmAnalyzedDistance = getDistance(d.get());
        std::optional<int> finalDistanceOrNoDependency;

        // getLevels reports the innermost loop. This parameter is currently not
        // used anywhere in Dynamatic (as of Mar. 4, 2026), but is extracted and
        // represented in the IR for completeness.
        unsigned depth = d->getLevels();
        if (/* When distance is not available (example: histogram) */
            !llvmAnalyzedDistance) {

          //
          // void histogram(in_int_t feature[1000], in_float_t weight[1000],
          //                inout_float_t hist[1000], in_int_t n) {
          //   for (int i = 0; i < n; ++i) {
          //     int m = feature[i];
          //     float wt = weight[i];
          //     float x = hist[m]; <-- LD
          //     hist[m + 3] = x + wt; <--- ST
          //   }
          // }
          //
          // In the histogram example above, the compiler identifies that src
          // has to go before dst in the inner-most loop in some cases, but it
          // cannot statically determine the exact distance (which depends on
          // the value of m). In this case, we can use the most conservative
          // order:
          //
          if (!src->comesBefore(dst)) {
            // CASE 1. If the reported distance is not the same as their
            // instruction sequence:
            //
            // --- program order --------
            // DST -> SRC
            // --- dependence reported --
            // (SRC, DST, "I don't know the distance")
            // --------------------------
            // here, we conservatively choose that the distance to be 1 (so the
            // ld in the next iteration already has to wait)
            //
            LLVM_DEBUG(llvm::errs() << "Dependence (distance unknown, "
                                       "conservatively set to 1):\n";
                       llvm::errs() << *src << "\n";
                       llvm::errs() << "  precedes:\n";
                       llvm::errs() << *dst << "\n";
                       llvm::errs() << "  distance: " << 1 << "\n";

            );
            finalDistanceOrNoDependency = 1;
          } else {
            // CASE 2. If the reported distance is the same as their instruction
            // sequence:
            //
            // --- program order --------
            // SRC -> DST
            // --- dependence reported --
            // (SRC, DST, "I don't know the distance")
            // --------------------------
            // here, we conservatively choose that the distance to be 0 (so the
            // st in the same iteration has to wait for the load).
            //
            LLVM_DEBUG(
                llvm::errs() << "Dependence (distance unknown, conservatively "
                                "set to 0):\n";
                llvm::errs() << *src << "\n"; llvm::errs() << "  precedes:\n";
                llvm::errs() << *dst << "\n";
                llvm::errs() << "  distance: " << 0 << "\n";);
            finalDistanceOrNoDependency = 0;
          }
        } else if (d->isOrdered()) {
          if (/* When distance is available and it is not a RAR (trivial dep.)
               */
              *llvmAnalyzedDistance == 0) {
            // Special case when dist == 0 (same BB): the analysis reports in
            // both directions. Here we we can directly use program order to
            // enforce:
            //
            // void test_memory_1(int a[N], int n) {
            //   for (int i = 0; i < n; i++)
            //     a[i] = a[i] + 5;
            // }
            // In this example, we technically cannot reorder the load/store
            // w.r.t their iterations, so the analysis tells us that there is
            // (RAW, 0) and (WAR, 0).
            //
            // In this case, we can use Instruction::comesBefore(...) to check
            // the actual program order.
            //
            if (src->comesBefore(dst)) {
              LLVM_DEBUG(
                  llvm::errs() << "Dependence (same iteration or no loop):\n";
                  llvm::errs() << *src << "\n"; llvm::errs() << "  precedes:\n";
                  llvm::errs() << *dst << "\n";
                  llvm::errs()
                  << "  distance: " << *llvmAnalyzedDistance << "\n";);
              finalDistanceOrNoDependency = *llvmAnalyzedDistance;
            }
          } else {
            LLVM_DEBUG(
                llvm::errs() << "Dependence:\n"; llvm::errs() << *src << "\n";
                llvm::errs() << "  precedes:\n"; llvm::errs() << *dst << "\n";
                llvm::errs()
                << "  distance: " << *llvmAnalyzedDistance << "\n";);
            finalDistanceOrNoDependency = *llvmAnalyzedDistance;
          }
        }

        if (
            // clang-format off
            /* there is a dependency */
            finalDistanceOrNoDependency &&
            /* it reports both fwd and bwd directions, we just need one of them for the SDC constraints */
            *finalDistanceOrNoDependency >= 0
            // clang-format on
        ) {
          if (instToDepsMap.count(src) == 0) {
            // This branch creates the list [dep1] if the predecessor
            // instruction hasn't been visited yet
            LLVMMemDependency newDep;
            newDep.name = nameMapping[src];
            newDep.destAndDepthAndDist.emplace_back(
                nameMapping[dst], depth, *finalDistanceOrNoDependency);
            instToDepsMap[src] = newDep;
          } else {
            // Otherwise, populate the existing list [dep1, dep2, ...] with the
            // new dep.
            instToDepsMap[src].destAndDepthAndDist.emplace_back(
                nameMapping[dst], depth, *finalDistanceOrNoDependency);
          }
        }

        LLVM_DEBUG(llvm::errs() << "-------------------------\n";);
      }
    }
  }
  llvm::LLVMContext &ctx = llvmFunction.getContext();

  for (auto [src, dests] : instToDepsMap) {
    dests.toLLVMMetaDataNode(ctx, src);
  }

  return PreservedAnalyses::all();
}

/// \brief: Refines `depMatrix` with the analysis inherited from the legacy
/// Dynamatic: Polly's polyhedral intersection of access relations, guided by
/// the GIID dataflow checks.
///
/// An access that no Scop covers is never touched, so its entries stay
/// unknown and are reported conservatively.
void MemDepAnalysisPass::refineWithPollyAnalysis(Function &llvmFunction,
                                                 FunctionAnalysisManager &fam,
                                                 DependenceMatrix &depMatrix) {

  auto &regionInfoAnalysis = fam.getResult<RegionInfoAnalysis>(llvmFunction);
  auto &scopInfoAnalysis = fam.getResult<ScopInfoAnalysis>(llvmFunction);
  auto &aliasAnalysis = fam.getResult<AAManager>(llvmFunction);

  std::deque<Region *> regionQueue;
  getAllRegions(*regionInfoAnalysis.getTopLevelRegion(), regionQueue);

  for (Region *region : regionQueue)
    if (Scop *scop = scopInfoAnalysis.getScop(region))
      processScop(*scop, depMatrix, aliasAnalysis);
}

PreservedAnalyses MemDepAnalysisPass::run(Function &llvmFunction,
                                          FunctionAnalysisManager &fam) {

  if (useDependenceAnalysis) {
    return this->runDependenceAnalysisBased(llvmFunction, fam);
  }

  llvm::LLVMContext &ctx = llvmFunction.getContext();

  auto nameMapping = nameAllLoadStores(llvmFunction);

  // The state of every ordered pair of memory accesses in the function. Every
  // entry starts out `Unknown`; each refinement below narrows the entries it
  // can prove, and whatever is still `Unknown` at the end is conservatively
  // treated as a real dependence.
  DependenceMatrix depMatrix(collectMemoryAccesses(llvmFunction));

  // Rule out the pairs that cannot be a dependence in the first place. These
  // are not analysis results. The order matters: see the note on
  // removeUnequalBase, which must come last of the three.
  removeEqual(depMatrix);
  removeRAR(depMatrix);
  removeUnequalBase(depMatrix);

  removeSucceedingPredecessor(depMatrix);
  removeNonAliasing(depMatrix, fam.getResult<AAManager>(llvmFunction),
                    fam.getResult<LoopAnalysis>(llvmFunction));
  refineWithPollyAnalysis(llvmFunction, fam, depMatrix);

  LLVM_DEBUG(depMatrix.print(llvm::dbgs()););

  // Group the edges by source access and serialize them onto the LLVM
  // instructions, where translate-llvm-to-std picks them up and turns them
  // into handshake::MemDependenceArrayAttr.
  for (auto [srcAccess, deps] : depMatrix.toDependencyMap(nameMapping)) {
    deps.toLLVMMetaDataNode(ctx, srcAccess);
  }

  return PreservedAnalyses::all();
}
} // namespace

// Register the pass for opt-style loading
// plugin:
// https://stackoverflow.com/questions/51474188/using-shared-object-so-by-command-opt-in-llvm
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "MemDepAnalysis", LLVM_VERSION_STRING,
          [](PassBuilder &pb) {
            pb.registerPipelineParsingCallback(
                [](StringRef name, FunctionPassManager &fpm,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (name == "mem-dep-analysis") {
                    fpm.addPass(MemDepAnalysisPass());
                    return true;
                  }
                  return false;
                });
          }};
}
