#include "polly/DependenceInfo.h"
#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"

#include "llvm/ADT/MapVector.h"
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

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Support/MemoryDependency.h"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/detail/adjacency_list.hpp>

using namespace llvm;
using namespace polly;

using Graph =
    boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>;
using Vertex = boost::graph_traits<Graph>::vertex_descriptor;
using Vertex_size_t = boost::graph_traits<Graph>::vertices_size_type;
using vertex_index_map =
    boost::property_map<Graph, boost::vertex_index_t>::const_type;

Value *findBaseInternal(Value *addr) {
  if (auto *arg = dyn_cast<Argument>(addr)) {
    if (!arg->getType()->isPointerTy())
      llvm_unreachable("Only pointer arguments are considered addresses");
    return addr;
  }

  if (isa<Constant>(addr))
    llvm_unreachable("Cannot determine base address of Constant");

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

namespace {

struct AccessInfo {
  std::map<Instruction *, isl::set> accessMaps;
  std::map<Instruction *, Value *> instToBase;

  std::map<Instruction *, unsigned> instToScopId;

  // Base to the set of instructions storing to this base
  std::map<Value *, std::set<Instruction *>> baseToInsts;

  bool sameScop(Instruction *i, Instruction *j) {
    if (!instToScopId.count(i))
      return false;

    if (!instToScopId.count(j))
      return false;

    return instToScopId.count(i) == instToScopId.count(j);
  }
};

struct ArrayPartition : PassInfoMixin<ArrayPartition> {

  unsigned memCount = 0;

  PreservedAnalyses run(Function &f, FunctionAnalysisManager &fam);
};

void getAllRegions(llvm::Region &r, std::deque<llvm::Region *> &rq) {
  rq.push_back(&r);
  for (const auto &e : r)
    getAllRegions(*e, rq);
}

PreservedAnalyses ArrayPartition::run(Function &f,
                                      FunctionAnalysisManager &fam) {

  llvm::LLVMContext &ctx = f.getContext();

  auto &regionInfoAnalysis = fam.getResult<RegionInfoAnalysis>(f);

  auto &scopInfoAnalysis = fam.getResult<ScopInfoAnalysis>(f);

  auto &loopAnalysis = fam.getResult<LoopAnalysis>(f);

  auto &aliasAnalysis = fam.getResult<AAManager>(f);

  llvm::errs() << "Hello!\n";

  AccessInfo info;

  std::deque<Region *> rq;
  getAllRegions(*regionInfoAnalysis.getTopLevelRegion(), rq);

  for (auto &bb : f) {
    for (auto &inst : bb) {
      if (isa<llvm::LoadInst, llvm::StoreInst>(&inst)) {
        Value *base = findBase(&inst);
        info.baseToInsts[base].insert(&inst);
      }
    }
  }

  Scop *s;
  unsigned scopId = 0;
  for (Region *r : rq) {
    if ((s = scopInfoAnalysis.getScop(r))) {
      for (auto &stmt : *s) {
        for (auto *memAccess : stmt) {
          auto *inst = memAccess->getAccessInstruction();

          info.instToScopId[inst] = scopId;

          // Find the access
          auto &memoryAccess = stmt.getArrayAccessFor(inst);

          // Maps iteration indices to array access indices:
          // example:
          // stmt[i, j] -> A[i, j]
          // stmt[i, j] -> B[i, j - 1]
          // NOTE:
          // - The base address might be different for different maps.
          isl::map currentMap = memoryAccess.getLatestAccessRelation();

          // The domain of the map, e.g., the loop bounds for the iterators.
          isl::set domain = stmt.getDomain();

          // The range of the map over the domain
          // e.g.,
          // - input: stmt[i, j] -> A[i, j] | i \in [0, N] and j \in [0, M]
          // - output: A[i, j] | i \in [0, N] and j \in [0, M]
          info.accessMaps[inst] = currentMap.intersect_domain(domain).range();
        }
      }
      scopId += 1;
    }
  }

  for (auto [base, insts] : info.baseToInsts) {

    // TODO: create graph

    for (Instruction *inst1 : insts) {
      for (Instruction *inst2 : insts) {

        if (inst1 == inst2)
          continue;

        bool isDependent = true;

        // If their are in the same scop
        if (info.sameScop(inst1, inst2)) {

          auto inst1Map = info.accessMaps[inst1];
          auto inst2Map = info.accessMaps[inst2];

          // If the two instructions might access the same index:
          isl::set intersect = inst1Map.intersect(inst2Map);

          isDependent = intersect.is_empty().is_false();
        } else {
          // Use the result from alias analysis to determine if the intructions
          // are dependent:
          // Otherwise, use results from alias analysis:
          AliasResult aliasResult = aliasAnalysis.alias(
              MemoryLocation::get(inst1), MemoryLocation::get(inst2));

          isDependent = aliasResult != AliasResult::NoAlias;
        }

        if (isDependent) {
          llvm::errs() << "Dependency found between: " << *inst1 << " and "
                       << *inst2 << "\n";
          info.instToBase[inst1] = base;
          info.instToBase[inst2] = base;
        } else {
          llvm::errs() << "No dependency between: " << *inst1 << " and "
                       << *inst2 << "\n";
        }
      }
    }
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
  return {LLVM_PLUGIN_API_VERSION, "ArrayPartition", LLVM_VERSION_STRING,
          [](PassBuilder &pb) {
            pb.registerPipelineParsingCallback(
                [](StringRef name, FunctionPassManager &fpm,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (name == "array-partition") {
                    fpm.addPass(ArrayPartition());
                    return true;
                  }
                  return false;
                });
          }};
}
