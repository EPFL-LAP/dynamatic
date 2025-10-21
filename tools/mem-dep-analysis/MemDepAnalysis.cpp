#include "polly/DependenceInfo.h"
#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include <stdexcept>
#include <stdlib.h>
#include <utility>

#include "llvm/Analysis/ValueTracking.h"

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Support/MemoryDependency.h"
#include "llvm/Analysis/ValueTracking.h"

using namespace llvm;
using namespace polly;

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
  bool hasGlobalInOrderInstrDependency(Instruction *dstInst,
                                       Instruction *srcInst);

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
      // different predecessors BB, so we store them in this map for now. We add
      // it to the Path.Val set before recursive calls
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

bool InstructionDependenceInfo::hasGlobalInOrderInstrDependency(
    Instruction *dstInst, Instruction *srcInst) {

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

} // namespace

using InstPairType = std::pair<Instruction *, Instruction *>;

/// \brief: An data container class that represents the analysis data from the
/// Scop.
/// https://www.cs.colostate.edu/~pouchet/software/polyopt/doc/htmltexinfo/Specifics-of-Polyhedral-Programs.html.
class ScopAnalysisInfo {
  LoopInfo *loopInfo;
  InstructionDependenceInfo instrDependenceInfo;

  int scopMinDepth;
  std::vector<Instruction *> memInsts;
  std::map<Instruction *, isl::map> instToCurrentMap;
  std::map<Instruction *, int> instToLoopDepth;
  std::set<InstPairType> intersections;
  std::map<Instruction *, llvm::Value *> instToBase;
  /// Each Minimized Scop has a separate context. This ensures that trying to
  /// intersect maps for instructions from separate Scops will raise an error
  isl::ctx ctx;
  // Used by the dependsInternal() function
  std::map<InstPairType, bool> dependsCache;
  std::set<InstPairType> outstandingDependsQueries;

  /// \brief (needs proof-read here): Find the loop depth of the outer most
  /// common loop that contain both instructions.
  int getOutMostCommonLoopDepth(Instruction *i0, Instruction *i1) {
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
  ScopAnalysisInfo(Scop &scop)
      : instrDependenceInfo(*scop.getLI()), ctx(isl::ctx(isl_ctx_alloc())) {
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
    auto *l = loopInfo->getLoopFor(bb);
    scopMinDepth = loopInfo->getLoopDepth(bb) - scop.getRelativeLoopDepth(l);
    assert(scopMinDepth > 0);
  }

  ~ScopAnalysisInfo() = default;

  /// \brief: Use addScopStmt() to add all ScopStmt's in a Scop. Then,
  /// computeIntersections() and finally getIntersectionList()
  void addScopStmt(ScopStmt &stmt) {
    int depth = loopInfo->getLoopDepth(stmt.getBasicBlock());

    for (auto *inst : stmt.getInstructions())
      if (inst->mayReadOrWriteMemory()) {
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

// \brief: This function computes the WAR and WAW dependencies in a Scop.
//
// \example:
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
// intersect entire store-set with entire load-set. For each write access,
// compare with relevant sets of read accesses.
//
// Similarly, two stores are checked for possible WAW conflicts

  // clang-format on
  void computeIntersections() {
    for (auto *storeInst : memInsts) {
      if (!storeInst->mayWriteToMemory())
        continue;

      // Checking for RAW and WAW conflicts between storeInst and secondInst
      for (auto *secondInst : memInsts) {
        /* Skip checking with self */
        if (secondInst == storeInst)
          continue;

        // No need to check between different arrays
        if (instToBase[secondInst] != instToBase[storeInst]) {
          continue;
        }

        int commonDepth = getOutMostCommonLoopDepth(secondInst, storeInst);

        auto pair = InstPairType(storeInst, secondInst);

        isl::map instMap, wrInstMap;

        // Condition:
        // - The store instruction has a GIID on secondInst (i.e., the
        // dependency of store on secondInst is **always** enforced by data
        // dependency).
        //
        // @Jiahui17: IMPORTANT NOTE: The original code also calls
        // "hasReverseDependency" (which tries to find storeInst starting from
        // secondInst instead). I think this is redundant to have both.
        //
        // Original code (I renamed hasTokenDependence to
        // hasGlobalInOrderInstrDependency):
        // https://github.com/lana555/dynamatic/blob/46f17ddcba58ffa77d33b988ab927f25abd6ab38/elastic-circuits/MemElemInfo/TokenDependenceInfo.cpp#L184-L211
        //
        bool hasDependency =
            instrDependenceInfo.hasGlobalInOrderInstrDependency(storeInst,
                                                                secondInst);

        auto *loadInst = dyn_cast_or_null<LoadInst>(secondInst);
        if (loadInst != nullptr && hasDependency) {
          // Consecutive top-level loops will finish the load before any store,
          // since there is an operand dependency.
          if (commonDepth == 0 && scopMinDepth == 1)
            continue;
          assert(commonDepth - scopMinDepth + 1 >= 0);
          unsigned depthToKeep = commonDepth - scopMinDepth + 1;
          instMap = getMap(secondInst, depthToKeep, true);
          wrInstMap = getMap(storeInst, depthToKeep, false);
        } else {
          // Generic case: we cannot put any restrictions on the indices being
          // processed by the instructions, if there are no token flow that can
          // be established between them. Therefore, we intersect the sets of
          // all possible indices ever accessed
          wrInstMap = getMap(storeInst, 0, false);
          instMap = getMap(secondInst, 0, false);
        }

        // If the two instructions might access the same index:
        isl::map intersect = instMap.intersect(wrInstMap);
        if (intersect.is_empty().is_false()) {
          intersections.insert(pair);
        }
      }
    }
  }

  std::set<InstPairType> &getIntersectionList() { return intersections; }

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
  std::set<InstPairType> instRAWlist;
  std::set<InstPairType> instWAWlist;
  std::set<BasicBlock *> bbList;
  std::map<BasicBlock *, int> bbToScopMap;
  std::map<Instruction *, Value *> instToBase;
};

void getAllRegions(llvm::Region &r, std::deque<llvm::Region *> &rq) {
  rq.push_back(&r);
  for (const auto &e : r)
    getAllRegions(*e, rq);
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
Value *findBaseInternal(Value *addr) {
  if (auto *arg = dyn_cast<Argument>(addr)) {
    if (!arg->getType()->isPointerTy())
      llvm_unreachable("Only pointer arguments are considered addresses");
    return addr;
  }

  // Example: returns a global constant or variable
  if (isa<Constant>(addr))
    return addr;

  if (auto *inst = dyn_cast_or_null<Instruction>(addr)) {
    if (isa<AllocaInst>(inst))
      return addr;
    if (auto *gepi = dyn_cast<GetElementPtrInst>(inst))
      return findBaseInternal(gepi->getPointerOperand());
    if (auto *si = dyn_cast<SelectInst>(inst)) {
      auto *trueBase = findBaseInternal(si->getTrueValue());
      auto *falseBase = findBaseInternal(si->getFalseValue());

      // Select must choose pointers to same array. Otherwise cannot
      // choose relevant arrayRAM in elastic circuit
      assert(trueBase == falseBase);
      return trueBase;
    }
  }

  // We try to find a few known cases of pointer expression. For others,
  // implement when you come across them
  llvm_unreachable("Cannot  determine base array, aborting...");
}

Value *findBase(Instruction *inst) {
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

namespace {

struct FunctionInfo {
  std::map<Instruction *, int> instToScopId;
  std::vector<Instruction *> loadInsts;
  std::vector<Instruction *> storeInsts;
  bool sameScop(Instruction *a, Instruction *b) const {
    if (instToScopId.count(a) == 0)
      return false;
    if (instToScopId.count(b) == 0)
      return false;
    return (instToScopId.at(a) == instToScopId.at(b));
  }
};

/// \brief: an LLVM pass that combines polyhedral and alias analysis to compute
/// a set of dependency edges from the LLVM IR. It further uses dataflow
/// analysis to eliminate dependency edges enforced by the dataflow.
struct MemDepAnalysisPass : PassInfoMixin<MemDepAnalysisPass> {

  IndexAnalysis indexAnalysis;
  AAManager::Result *aliasAnalysis;
  unsigned memCount = 0;

  /// \brief: Loops through the scop regions in the IR and applies index and
  /// dataflow analysis to compute the minimum set of dependency edges.
  void processScop(Scop &s, std::vector<ScopAnalysisInfo> &scopMeta);

  /// \brief: Loops through the loops in the IR and collect the loads and
  /// stores.
  void processLoop(Loop *l, std::vector<struct LoopMetaData> &loopMetaInfos);
  PreservedAnalyses run(Function &f, FunctionAnalysisManager &fam);

  /// \brief: returns a list of (srcInst, dstInst) pairs that might have a WAR
  /// or WAW conflict.
  std::vector<InstPairType>
  getDependencyPairs(const FunctionInfo &functionInfo);
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

void MemDepAnalysisPass::processScop(Scop &scop,
                                     std::vector<ScopAnalysisInfo> &scopMeta) {

  auto meta = ScopAnalysisInfo(scop);

  for (auto &stmt : scop) {
    auto *bb = stmt.getBasicBlock();
    indexAnalysis.bbList.insert(bb);
    indexAnalysis.bbToScopMap[bb] = scopMeta.size();

    if (!hasMemoryReadOrWrite(stmt))
      continue;

    meta.addScopStmt(stmt);
  }

  meta.computeIntersections();

  for (auto [i, v] : meta.getInstsToBase()) {
    indexAnalysis.instToBase[i] = v;
  }

  for (auto pair : meta.getIntersectionList()) {
    // The convention used in ScopMeta class is that the first element in an
    // instPair is a store instruction. Thus, checking the type of the second
    // instruction tells us whther it is a RAW/WAW dependency
    if (pair.second->mayWriteToMemory())
      indexAnalysis.instWAWlist.insert(pair);
    else
      indexAnalysis.instRAWlist.insert(pair);
  }

  scopMeta.push_back(meta);
}

std::vector<InstPairType>
MemDepAnalysisPass::getDependencyPairs(const FunctionInfo &functionInfo) {
  std::vector<InstPairType> depPairList;
  for (auto *storeInst : functionInfo.storeInsts) {
    // Find RAW dependencies
    for (auto *loadInst : functionInfo.loadInsts) {

      InstPairType rawPair = std::make_pair(storeInst, loadInst);

      // NOTE: In dynamatic we assume that memory with different base addresses
      // are store in separate RAMs. Two instructions targetting differing base
      // arrays can never conflict.
      if (!equalBase(storeInst, loadInst))
        continue;

      // Instructions are in the same scop: use the result from IndexAnalysis
      if (functionInfo.sameScop(loadInst, storeInst)) {
        if (indexAnalysis.instRAWlist.count(rawPair) > 0)
          depPairList.push_back(rawPair);
        continue;
      }

      // Instruction are in different Scops: use the result from alias analysis
      AliasResult aliasResult = aliasAnalysis->alias(
          MemoryLocation::get(loadInst), MemoryLocation::get(storeInst));

      // If they always or sometimes alias:
      if (aliasResult != AliasResult::NoAlias) {
        depPairList.push_back(rawPair);
      }
    }
    // Find WAW dependencies
    for (auto *secondStoreInst : functionInfo.storeInsts) {
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
      if (functionInfo.sameScop(storeInst, secondStoreInst)) {
        if (indexAnalysis.instWAWlist.count(pair) > 0)
          depPairList.push_back(pair);
        else if (indexAnalysis.instWAWlist.count(pairRev) > 0)
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

PreservedAnalyses MemDepAnalysisPass::run(Function &f,
                                          FunctionAnalysisManager &fam) {

  llvm::LLVMContext &ctx = f.getContext();

  auto &regionInfoAnalysis = fam.getResult<RegionInfoAnalysis>(f);

  auto &scopInfoAnalysis = fam.getResult<ScopInfoAnalysis>(f);

  std::vector<ScopAnalysisInfo> scopMetaInfos;

  aliasAnalysis = &fam.getResult<AAManager>(f);

  std::deque<Region *> rq;
  getAllRegions(*regionInfoAnalysis.getTopLevelRegion(), rq);

  Scop *s;
  for (Region *r : rq) {
    if ((s = scopInfoAnalysis.getScop(r)))
      processScop(*s, scopMetaInfos);
  }

  FunctionInfo functionInfo;

  for (auto &bb : f) {
    int scopId = indexAnalysis.getScopID(&bb);
    bool isInScop = indexAnalysis.isInScop(&bb);
    for (auto &inst : bb) {
      if (!inst.mayReadOrWriteMemory())
        continue;
      if (isa<CallInst>(&inst)) {
        llvm::errs() << "Warning - Applying memory analysis on a function with "
                        "a call instruction!\n";
        continue;
      }

      // NOTE: In legacy dynamatic here uses mayReadFromMemory and
      // mayWriteToMemory, which I think is quite redundant for our need
      if (isa<llvm::LoadInst>(&inst))
        functionInfo.loadInsts.emplace_back(&inst);
      if (isa<llvm::StoreInst>(&inst))
        functionInfo.storeInsts.emplace_back(&inst);
      if (isInScop)
        functionInfo.instToScopId[&inst] = scopId;
    }
  }

  auto nameMapping = nameAllLoadStores(f);

  std::map<Instruction *, LLVMMemDependency> deps;
  for (auto &[src, dst] : getDependencyPairs(functionInfo)) {
    assert(nameMapping.count(src) > 0 && "Unnamed load/store op!");
    if (deps.count(src) == 0) {
      LLVMMemDependency newDep;
      newDep.name = nameMapping[src];
      newDep.destAndDepth.emplace_back(nameMapping[dst], 1);
      deps[src] = newDep;
    } else {
      deps[src].destAndDepth.emplace_back(nameMapping[dst], 1);
    }
  }

  for (auto [src, dests] : deps) {
    dests.toLLVMMetaDataNode(ctx, src);
  }

  return PreservedAnalyses::all();
}

} // end anonymous namespace

// Register the pass for opt-style loading
// Important note: you need to enable shared libarary in LLVM to load pass
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
