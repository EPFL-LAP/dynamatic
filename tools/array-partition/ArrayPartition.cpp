#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"
#include "polly/Support/ISLTools.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "isl/point.h"
#include "isl/set.h"
#include <boost/graph/connected_components.hpp>
#include <boost/property_map/property_map.hpp>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <stdlib.h>
#include <utility>

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Support/MemoryDependency.h"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/detail/adjacency_list.hpp>

#include "llvm/Support/Debug.h"
#include <polly/Support/ISLOStream.h>

#define DEBUG_TYPE "array-parition"

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

  if (isa<Constant>(addr)) {
    // Example: This can be a global constant array:
    // @w0 = dso_local constant [64 x [16 x i32]] ... (values)
    return addr;
  }

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
void changeGEPOperands(Instruction *gepInst, Value *newBasePtr,
                       Type *newArrayType, const ArraySquashingInfo &info) {

  // NOTE: This is a special case where findBaseGEP returns a load. This happens
  // when we access Array[0][0]...[0]. In this case, we do not need to update
  // the address calculation (because firstIndex must be 0, so "(idx -
  // firstIndex) / step" must still be zero). We just need to update the address
  if (auto *loadInst = dyn_cast<LoadInst>(gepInst)) {
    if (loadInst->getPointerOperand() != newBasePtr) {
      loadInst->setOperand(0, newBasePtr);
    }
    return;
  }

  if (auto *storeInst = dyn_cast<StoreInst>(gepInst)) {
    if (storeInst->getPointerOperand() != newBasePtr) {
      storeInst->setOperand(1, newBasePtr);
    }
    return;
  }

  if (!isa<GetElementPtrInst>(gepInst)) {
    gepInst->dump();
    llvm_unreachable(
        "Trying to change the operands of an operation with unhandled type.");
  }

  auto *gep = dyn_cast<GetElementPtrInst>(gepInst);
  if (gep->getPointerOperand() != newBasePtr) {
    // Change the base pointer of the GEP instruction
    gep->setOperand(0, newBasePtr);
  }

  gep->setSourceElementType(newArrayType);

  // NOTE: Both GEP and info store descreasing dimensions.
  // Example: A[3][4][5]
  // - We iterate through 3 -> 4 -> 5
  for (unsigned i = 0; i < info.size(); i++) {
    auto [firstIndex, step, elems] = info[i];

    // The GEP indices have an extra preceeding zero index, here we skip it if
    // it is the case (i.e., gep->getNumIndices() == info.size() + 1)
    // Source:
    // https://llvm.org/docs/GetElementPtr.html#why-is-the-extra-0-index-required
    auto *indexOprd =
        gep->idx_begin() + i + (gep->getNumIndices() - info.size());
    if (i < gep->getNumOperands() - 1) {
      // If we have enough indices, change the index
      auto *index = (*indexOprd).get();
      if (auto *constInt = dyn_cast<ConstantInt>(index)) {
        int64_t oldIdx = constInt->getSExtValue();
        int64_t newIdx = (oldIdx - firstIndex) / step;
        *indexOprd = ConstantInt::get(constInt->getType(), newIdx);
      } else if (auto *gepIndex = dyn_cast<Value>(index)) {

        IRBuilder<> builder(gep);
        auto *subOutput = builder.CreateSub(
            gepIndex, ConstantInt::get(gepIndex->getType(), firstIndex));
        auto *divOutput = builder.CreateUDiv(
            subOutput, ConstantInt::get(gepIndex->getType(), step));
        *indexOprd = divOutput;
      } else {
        llvm_unreachable("GEP index is not a constant integer");
      }
    }
  }
}

namespace {

struct AccessInfo {
  std::map<Instruction *, isl::set> accessMaps;

  std::map<Instruction *, unsigned> instToScopId;

  // Base to the set of instructions storing to this base
  std::map<Value *, std::set<Instruction *>> baseToInsts;

  bool sameScop(Instruction *i, Instruction *j) const {
    if (!instToScopId.count(i))
      return false;

    if (!instToScopId.count(j))
      return false;

    return instToScopId.count(i) == instToScopId.count(j);
  }
};

/// \note: It returns ArraySquashingInfo info. info[0] gives the information of
/// the outer-most dimension (same convention as GEP: from outer to inner)
ArraySquashingInfo extractDimInfo(const isl::set &range,
                                  llvm::Type *allocaElemType) {

  LLVM_DEBUG(llvm::errs() << "Extracting dimension info for access range "
                          << range << "\n";);

  ArraySquashingInfo info;

  if (range.is_null()) {
    // NULL range: this means that some accesses are not in the Scop and we
    // don't know the access range of them. Here we simply return a full index
    // range to be on the safe side.
    while (allocaElemType->isArrayTy()) {
      info.emplace_back(0, 1, allocaElemType->getArrayNumElements());
      allocaElemType = allocaElemType->getArrayElementType();
    }
    return info;
  }

  // example: A[N][M] gives you 2 dimensions
  auto numDims = unsignedFromIslSize(range.as_set().dim(isl::dim::set));

  // For each dimension: enumerate all reachable indices
  for (unsigned i = 0; i < numDims; i++) {
    assert(allocaElemType->isArrayTy() &&
           "The allocated element type is not an array (e.g., the global array "
           "is only paritially intialized)?");
    auto originalDimSize = allocaElemType->getArrayNumElements();

    isl::set reachableIndicesIslSet = range;
    for (unsigned j = 0; j < numDims; ++j) {
      // We need to remove all other dimensions that are not "i".
      if (i != j) {
        // "project_out" existence-quantifies a range of specified dimensions.
        reachableIndicesIslSet =
            range.project_out(isl::dim::set,
                              /* starting from which dimension? */ j,
                              /* how many dimensions? */ 1);
      }
    }

    // The reachableIndicesIslSet is an isl::set type; we convert it to a set of
    // integers to compute the step size and start index.
    std::vector<int> reachableIndices;
    reachableIndicesIslSet.foreach_point(
        [&reachableIndices](const isl::point &p) {
          // isl::point is a multidimensional vector (a, b, c, d).
          // "coordinate_val" retrives the values of pos-th dimension.
          // In this case, dim only has one dimension (other dimensions are
          // quantified away) so the position of the dimension is always 0.
          auto val = p.coordinate_val(isl::dim::set, 0);
          int actualVal = val.get_num_si();
          reachableIndices.push_back(actualVal);
          return isl::stat::ok();
        });

    assert(reachableIndices.size() <= originalDimSize &&
           "The number of reachable indices should not exceed the original "
           "array size!");
    std::sort(reachableIndices.begin(), reachableIndices.end());
    std::set<int> diffs;
    for (size_t i = 0; i + 1 < reachableIndices.size(); ++i) {
      int diff = reachableIndices[i] - reachableIndices[i + 1];
      diffs.insert(diff);
    }

    if (diffs.size() != 1) {
      info.emplace_back(0,
                        /* step = 1 indicates that we can't squash the array
                           into a smaller one currently */
                        1, originalDimSize);
    } else {
      info.emplace_back(reachableIndices.front(), abs(*diffs.begin()),
                        reachableIndices.size());
    }
    allocaElemType = allocaElemType->getArrayElementType();
  }

  return info;
}

/// \brief: Construct the type needed for the new allocaOp
///
/// \note: dims is the number of elements in each dimension. Caveat:
/// - suppose that you declared A[3][4][5], then dims should be {5, 4, 3}
llvm::Type *getAllocaElemType(Type *baseElementType,
                              const ArraySquashingInfo &dims) {
  Type *elemTy = baseElementType;
  // Reverse because dim is from outer to inner, but here the construction is
  // from inner to outer
  for (auto [init, step, elems] : llvm::reverse(dims)) {
    elemTy = ArrayType::get(elemTy, elems);
  }
  return elemTy;
}

struct ArrayPartition : PassInfoMixin<ArrayPartition> {

  unsigned memCount = 0;

  PreservedAnalyses run(Function &f, FunctionAnalysisManager &fam);
};

AllocaInst *createAlloca(AllocaInst *origAlloca,
                         const ArraySquashingInfo &info) {
  Instruction *insertPoint = origAlloca->getNextNode();
  IRBuilder<> builder(insertPoint);

  Type *baseElementType = origAlloca->getAllocatedType();

  while (baseElementType->isArrayTy()) {
    baseElementType = baseElementType->getArrayElementType();
  }

  Type *allocatedType = getAllocaElemType(baseElementType, info);
  Value *arraySize = origAlloca->getArraySize();
  Align alignment = origAlloca->getAlign();

  // Create new alloca and let LLVM assign a unique name
  AllocaInst *newAlloca =
      builder.CreateAlloca(allocatedType, arraySize, origAlloca->getName());
  newAlloca->setAlignment(alignment);

  return newAlloca;
}

void getAllRegions(llvm::Region &r, std::deque<llvm::Region *> &rq) {
  rq.push_back(&r);
  for (const auto &e : r)
    getAllRegions(*e, rq);
}

/// The memory accesses are grouped together. Between the groups there are no
/// overlapping accesses.
///
/// \example: InstsPerGroup groups;
/// groups[1] returns all the instructions in group 1
using InstsPerGroup = std::vector<std::set<Instruction *>>;
InstsPerGroup computeInstsPerGroup(const std::set<Instruction *> &setOfInsts,
                                   AccessInfo &info,
                                   AAManager::Result &aliasAnalysis) {

  std::vector<Instruction *> insts(setOfInsts.begin(), setOfInsts.end());

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
  size_t numComponents = boost::connected_components(g, &nodeToComponentId[0]);
  InstsPerGroup groups(numComponents);
  for (size_t i = 0; i < insts.size(); ++i) {
    size_t compId = nodeToComponentId[i];
    groups[compId].insert(insts[i]);
  }

  return groups;
}

void partitionVariableAlloca(llvm::AllocaInst *baseAlloca,
                             std::set<Instruction *> &insts, AccessInfo &info,
                             AAManager::Result &aliasAnalysis,
                             isl::ctx islCtx) {
  auto groups = computeInstsPerGroup(insts, info, aliasAnalysis);

  if (groups.size() == 1) {
    // Cannot partition the array: every memory instruction is conflicting
    // with another one
    return;
  }

  for (auto &group : groups) {
    // Make an empty set (note: somehow if you just do "isl::union_set
    // range;" it wouldn't work)
    isl::union_set range = isl::union_set::empty(islCtx);

    // This computes the union of all memory access indices in the group
    for (auto *inst : group) {
      auto instRange = info.accessMaps[inst];
      range = range.unite(instRange);
    }

    auto dimInfo =
        extractDimInfo(range.as_set(), baseAlloca->getAllocatedType());

    auto *newAlloca = createAlloca(baseAlloca, dimInfo);

    for (auto *inst : group) {
      auto *gepBase = findBaseGEP(inst);
      changeGEPOperands(gepBase, newAlloca, newAlloca->getAllocatedType(),
                        dimInfo);
    }
  }
}

llvm::Constant *
getElementFromGlobalArray(llvm::GlobalVariable *globVar,
                          const std::vector<unsigned> &indices) {

  if (!globVar->hasInitializer()) {
    llvm::errs() << "Global variable does not have an initializer: "
                 << globVar->getName() << "\n";
    return nullptr;
  }

  llvm::Constant *init = globVar->getInitializer();
  for (auto idx : llvm::drop_end(indices)) {
    auto *array = llvm::dyn_cast<llvm::ConstantArray>(init);

    if (!array) {
      llvm::errs() << "Expected a constant array, but got: "
                   << init->getType()->getTypeID() << "\n";
      return nullptr;
    }

    if (idx >= array->getNumOperands()) {
      llvm::errs() << "Invalid index " << idx << " for array\n";
      return nullptr;
    }
    init = array->getOperand(idx);
  }

  auto *array = llvm::dyn_cast<llvm::ConstantDataArray>(init);

  unsigned idx = indices.back();

  if (!array) {
    llvm::errs() << "Expected a constant data array, but got: "
                 << init->getType()->getTypeID() << "\n";
    return nullptr;
  }

  if (idx >= array->getNumElements()) {
    llvm::errs() << "Invalid index " << idx << " for constant data array\n";
    return nullptr;
  }

  init = array->getElementAsConstant(idx);

  return init;
}

llvm::Constant *constructGlobalConstantTensor(
    ArraySquashingInfo &info, const std::vector<unsigned> &indices,
    llvm::GlobalVariable *originalGbl, unsigned dims) {

  // The inner most dimension (i.e., this give a scalar value)
  if (indices.size() == dims) {
    return getElementFromGlobalArray(originalGbl, indices);
  }

  // Try to iterate through the current dimension, and make a new array constant

  std::vector<llvm::Constant *> newArray;

  // Iterate through the current dimension
  auto &[firstIdx, step, elems] = info[indices.size()];

  for (unsigned i = 0; i < elems; ++i) {
    // Construct the new indices
    std::vector<unsigned> newIndices = indices;
    newIndices.push_back(firstIdx + step * i);

    // Get the element from the original global variable
    auto *element =
        constructGlobalConstantTensor(info, newIndices, originalGbl, dims);
    assert(element);
    newArray.push_back(element);
  }

  Type *arrayType =
      llvm::ArrayType::get(newArray.front()->getType(), newArray.size());

  return llvm::ConstantArray::get(llvm::cast<llvm::ArrayType>(arrayType),
                                  newArray);
}

void partitionGlobalAlloca(Module *mod, llvm::GlobalVariable *gblConstant,
                           std::set<Instruction *> &insts, AccessInfo &info,
                           AAManager::Result &aliasAnalysis, isl::ctx islCtx

) {
  if (!gblConstant->hasInitializer())
    return;

  auto groups = computeInstsPerGroup(insts, info, aliasAnalysis);

  if (groups.size() == 1) {
    // Cannot partition the array: every memory instruction is conflicting
    // with another one
    return;
  }

  for (auto &group : groups) {
    // Make an empty set (note: somehow if you just do "isl::union_set
    // range;" it wouldn't work)
    isl::union_set range = isl::union_set::empty(islCtx);

    // This computes the union of all memory access indices in the group
    for (auto *inst : group) {
      auto instRange = info.accessMaps[inst];
      range = range.unite(instRange);
    }
    auto dimInfo = extractDimInfo(range.as_set(), gblConstant->getValueType());
    // Get all the memory values accessed in the array:

    auto *constArray =
        constructGlobalConstantTensor(dimInfo, {}, gblConstant, dimInfo.size());

    auto *gVar = new llvm::GlobalVariable(
        *mod, constArray->getType(),
        /*isConstant=*/true, llvm::GlobalValue::InternalLinkage, constArray,
        gblConstant->getName() + "duplicated");
    gVar->setAlignment(gblConstant->getAlign());
    for (auto *inst : group) {
      auto *gepBase = findBaseGEP(inst);
      changeGEPOperands(gepBase, gVar, gVar->getValueType(), dimInfo);
    }
  }
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

  // Needed for constructing the global constants
  Module *mod = f.getParent();

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
          auto *memoryAccess = stmt.getArrayAccessOrNULLFor(inst);

          if (!memoryAccess) {
            continue;
          }

          // Maps iteration indices to array access indices:
          // example:
          // stmt[i, j] -> A[i, j]
          // stmt[i, j] -> B[i, j - 1]
          // NOTE:
          // - The base address might be different for different maps.
          isl::map currentMap = memoryAccess->getLatestAccessRelation();

          // The domain of the map, e.g., the loop bounds for the iterators.
          isl::set domain = stmt.getDomain();

          // The range of the map over the domain
          // e.g.,
          // - input: stmt[i, j] -> A[i, j] | i \in [0, N] and j \in [0, M]
          // - output: A[i, j] | i \in [0, N] and j \in [0, M]
          isl::set range = currentMap.intersect_domain(domain).range();
          info.accessMaps[inst] = range;
        }
      }
      scopId += 1;
    }
  }

  // For each base address of the GEPs that are allocas (i.e., a separate RAM in
  // HLS circuit), compute the optimal partitioning and create separate alloca
  // instructions.
  for (auto [base, insts] : info.baseToInsts) {
    if (isa<Instruction>(base)) {
      if (auto *allocaInst = dyn_cast<AllocaInst>(base)) {
        // If the base is an alloca, we can partition it
        partitionVariableAlloca(allocaInst, insts, info, aliasAnalysis, islCtx);
      } else {
        assert(false &&
               "Base address of the GEP is not an alloca, cannot partition!");
      }
    }

    if (isa<Constant>(base)) {
      if (auto *globVar = dyn_cast<GlobalVariable>(base)) {
        // If the base is a global variable, we can partition it
        partitionGlobalAlloca(mod, globVar, insts, info, aliasAnalysis, islCtx);
      }
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
