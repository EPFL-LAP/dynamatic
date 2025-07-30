#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"
#include "polly/Support/ISLTools.h"

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
#include "isl/point.h"
#include <boost/graph/connected_components.hpp>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <stdlib.h>
#include <utility>

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Support/MemoryDependency.h"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/detail/adjacency_list.hpp>

#include <boost/throw_exception.hpp>
void boost::throw_exception(std::exception const &e) { std::abort(); }

using namespace llvm;
using namespace polly;

using Graph =
    boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>;
using Vertex = boost::graph_traits<Graph>::vertex_descriptor;
using Vertex_size_t = boost::graph_traits<Graph>::vertices_size_type;
using vertex_index_map =
    boost::property_map<Graph, boost::vertex_index_t>::const_type;

/// \brief: Analysis information for reducing the array size if the accesses
/// range is affine
///
/// \example: suppose you have an array v[10], and you only access the old
/// number: [1, 3, 5, 7, 9]. This holds the information about what indices are
/// accessed.
/// (firstIndex, stepSize, numElements). In this case, we can redefine the array
/// as v[1].
///
/// After redefining the array, we still need to update the GEP to point to the
/// new addresses.
/// We assume that the GEP is already instcombined.
///
/// The access index of the new GEP would be (oldIdx - firstIndex) / stepSize.
using DimInfo = std::tuple<unsigned, unsigned, unsigned>;
using ArraySquashingInfo = std::vector<DimInfo>;

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

Instruction *findBaseGEPInternal(Value *addr, Instruction *last) {
  if (!isa<GetElementPtrInst>(addr)) {
    return last;
  }
  auto *gep = cast<GetElementPtrInst>(addr);
  return findBaseGEPInternal(gep->getPointerOperand(), gep);
}

Instruction *findBaseGEP(Instruction *inst) {
  Value *addr;
  if (auto *loadInst = dyn_cast<LoadInst>(inst)) {
    addr = loadInst->getPointerOperand();
  } else if (auto *storeInst = dyn_cast<StoreInst>(inst)) {
    addr = storeInst->getPointerOperand();
  } else {
    llvm_unreachable("Instruction is not a memory access");
  }

  return findBaseGEPInternal(addr, inst);
}

// FIXME: for now, we only duplicate the alloca instruction. We could optimize
// away certain locations that are never used.
AllocaInst *cloneAllocaAfter(AllocaInst *origAlloca) {
  Instruction *insertPoint = origAlloca->getNextNode();
  IRBuilder<> builder(insertPoint);

  Type *allocatedType = origAlloca->getAllocatedType();
  Value *arraySize = origAlloca->getArraySize();
  Align alignment = origAlloca->getAlign();

  // Create new alloca and let LLVM assign a unique name
  AllocaInst *newAlloca = builder.CreateAlloca(allocatedType, arraySize);
  newAlloca->setName(origAlloca->getName() + ".cloned");
  newAlloca->setAlignment(alignment);

  return newAlloca;
}

void changeGEPBasePtr(Instruction *gepInst, Value *newBasePtr) {
  auto *gep = cast<GetElementPtrInst>(gepInst);
  if (gep->getPointerOperand() != newBasePtr) {
    // Change the base pointer of the GEP instruction
    gep->setOperand(0, newBasePtr);
  }
}

/// \brief: After shrinking the array to an optimal size, we also need to update
/// the GEP ops to the new array.
///
/// 1. Change the arrayType
/// 2. Change the indices to point to the new array
///
/// Note: this pass assumes that the GEPs are already instcombined. So one GEP
/// manages the indexing of all dimensions.
void changeGEPOperands(Instruction *gep, Value *newArrayType,
                       const ArraySquashingInfo &info) {

  //
}

namespace {

struct AccessInfo {
  std::map<Instruction *, isl::set> accessMaps;

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

ArraySquashingInfo extractDimInfo(isl::set range) {
  ArraySquashingInfo info;

  llvm::errs() << "Enumerating points! with num dims\n";

  // example: A[N][M] gives you 2 dimensions
  auto numDims = unsignedFromIslSize(range.as_set().dim(isl::dim::set));

  for (unsigned i = 0; i < numDims; i++) {
    auto dim0 = range.as_set().project_out(
        isl::dim::set, /*starting from which dimension? */ i,
        /* how many dimensions? */ 1);

    std::vector<int> reachableIndices;
    dim0.foreach_point([&reachableIndices](isl::point p) {
      auto val = p.coordinate_val(
          isl::dim::set,
          /* dim only has one dimension so the position of the dim is 0 */
          0);

      int actualVal = val.get_num_si();
      reachableIndices.push_back(actualVal);
      return isl::stat::ok();
    });
    llvm::errs() << "Number of indices! " << reachableIndices.size() << "\n";
    std::sort(reachableIndices.begin(), reachableIndices.end());
    std::set<int> diffs;
    for (size_t i = 0; i + 1 < reachableIndices.size(); ++i) {
      int diff = reachableIndices[i] - reachableIndices[i + 1];
      diffs.insert(diff);
    }

    if (diffs.size() != 1) {
      info.emplace_back(reachableIndices.front(),
                        /* step = 1 indicates that we can't squash the array
                           into a smaller one currently */
                        1, reachableIndices.size());
    } else {
      info.emplace_back(reachableIndices.front(), abs(*diffs.begin()),
                        reachableIndices.size());
    }
  }

  return info;
}

/// \brief: Construct the type needed for the new allocaOp
///
/// \note: dims is the number of elements in each dimension. Caveat:
/// - suppose that you declared A[3][4][5], then dims should be {5, 4, 3}
llvm::Type *getAllocaElemType(Type *baseElementType,
                              const ArraySquashingInfo &dims) {
  Type *ElemTy = baseElementType;
  for (auto [init, step, elems] : dims) {
    ElemTy = ArrayType::get(ElemTy, elems);
  }
  return ElemTy;
}

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

  if (f.getName() == "main") {
    llvm::errs()
        << "Skipping main function for automatic array partitioning!\n";
    return PreservedAnalyses::all();
  }

  auto islCtx = isl::ctx(isl_ctx_alloc());

  auto &regionInfoAnalysis = fam.getResult<RegionInfoAnalysis>(f);

  auto &scopInfoAnalysis = fam.getResult<ScopInfoAnalysis>(f);

  auto &aliasAnalysis = fam.getResult<AAManager>(f);

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

    if (!isa<Instruction>(base)) {
      continue;
    }

    auto *baseAlloca = cast<AllocaInst>(base);
    if (!baseAlloca) {
      continue;
    }

    llvm::errs() << "Base alloca: " << *baseAlloca << "\n";

    std::map<Instruction *, Vertex> instToVertex;
    std::map<Vertex, Instruction *> vertexToInst;
    Graph g;

    for (Instruction *inst : insts) {
      Vertex v = boost::add_vertex(g);
      instToVertex[inst] = v;
      vertexToInst[v] = inst;
    }

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
          // Use the result from alias analysis to determine if the
          // intructions are dependent: Otherwise, use results from alias
          // analysis:
          AliasResult aliasResult = aliasAnalysis.alias(
              MemoryLocation::get(inst1), MemoryLocation::get(inst2));

          isDependent = aliasResult != AliasResult::NoAlias;
        }

        if (isDependent) {

          auto v1 = instToVertex[inst1];
          auto v2 = instToVertex[inst2];
          boost::add_edge(v1, v2, g);
        }
      }
    }
    // Find the connected components in the graph:
    std::vector<int> nodeToComponentId(boost::num_vertices(g),
                                       /* -1 : not assigned (error) */ -1);
    size_t numComponents =
        boost::connected_components(g, &nodeToComponentId[0]);

    for (size_t i = 1; i < numComponents; i++) {
      // Make an empty set (note: somehow if you just do "isl::union_set
      // range;" it wouldn't work)
      isl::union_set range = isl::union_set::empty(islCtx);
      for (size_t j = 0; j < nodeToComponentId.size(); j++) {
        if (nodeToComponentId[j] == static_cast<int>(i)) {
          auto *inst = vertexToInst[boost::vertex(j, g)];
          auto instRange = info.accessMaps[inst];
          range = range.unite(instRange);
        }
      }
      // Enumerate points

      llvm::errs() << "Enumerating points! with num dims\n";

      auto dimInfo = extractDimInfo(range.as_set());

      llvm::errs() << "Creating new alloca to improve parallelism...\b";
      // Make a new alloca
      // FIXME: we can optimize this by squashing the unused locations in this
      // memory
      auto *newAlloca = cloneAllocaAfter(baseAlloca);

      for (size_t j = 0; j < nodeToComponentId.size(); j++) {
        if (nodeToComponentId[j] == static_cast<int>(i)) {
          changeGEPBasePtr(findBaseGEP(vertexToInst[boost::vertex(j, g)]),
                           newAlloca);
        }
      }
      llvm::errs() << "\n";
    }
  }
  return PreservedAnalyses::all();
}

} // namespace

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
