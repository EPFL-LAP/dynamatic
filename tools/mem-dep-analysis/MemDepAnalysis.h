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

using InstPairType = std::pair<llvm::Instruction *, llvm::Instruction *>;

class ScopAnalysisInfo;

struct FunctionInfo {
  std::map<llvm::Instruction *, int> instToScopId;
  std::vector<llvm::Instruction *> loadInsts;
  std::vector<llvm::Instruction *> storeInsts;
  bool sameScop(llvm::Instruction *a, llvm::Instruction *b) const {
    if (instToScopId.count(a) == 0)
      return false;
    if (instToScopId.count(b) == 0)
      return false;
    return (instToScopId.at(a) == instToScopId.at(b));
  }
};

struct IndexAnalysis {

  IndexAnalysis() : otherInsts() {}
  ~IndexAnalysis() = default;

  /// Returns all memory instructions in SCoPs which do not require an LSQ
  /// connection
  std::vector<llvm::Instruction *> &getOtherInsts() { return otherInsts; }

  /// Query whether any SCoP contains BB
  bool isInScop(llvm::BasicBlock *bb) {
    return bbList.find(bb) != bbList.end();
  }

  /// Returns an integer uniquely identifying the SCoP which contains BB
  int getScopID(llvm::BasicBlock *bb) {
    return (isInScop(bb)) ? bbToScopMap[bb] : -1;
  }

  std::vector<llvm::Instruction *> otherInsts;
  std::set<InstPairType> instRAWlist;
  std::set<InstPairType> instWAWlist;
  std::set<llvm::BasicBlock *> bbList;
  std::map<llvm::BasicBlock *, int> bbToScopMap;
  std::map<llvm::Instruction *, llvm::Value *> instToBase;
};

/// \brief: an LLVM pass that combines polyhedral and alias analysis to compute
/// a set of dependency edges from the LLVM IR. It further uses dataflow
/// analysis to eliminate dependency edges enforced by the dataflow.
struct MemDepAnalysisPass : llvm::PassInfoMixin<MemDepAnalysisPass> {

  IndexAnalysis indexAnalysis;
  llvm::AAManager::Result *aliasAnalysis;
  unsigned memCount = 0;

  /// \brief: Loops through the scop regions in the IR and applies index and
  /// dataflow analysis to compute the minimum set of dependency edges.
  void processScop(llvm::Scop &s, std::vector<ScopAnalysisInfo> &scopMeta);

  /// \brief: Loops through the loops in the IR and collect the loads and
  /// stores.
  void processLoop(llvm::Loop *l,
                   std::vector<struct LoopMetaData> &loopMetaInfos);
  llvm::PreservedAnalyses run(llvm::Function &f,
                              llvm::FunctionAnalysisManager &fam);

  /// \brief: returns a list of (srcInst, dstInst) pairs that might have a WAR
  /// or WAW conflict.
  std::vector<InstPairType>
  getDependencyPairs(const FunctionInfo &functionInfo);
  std::map<llvm::Instruction *, std::string>
  nameAllLoadStores(llvm::Function &f);
};
