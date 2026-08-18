// -----------------------------------------------------------------------
// Automatically partition the array using polyhedral analysis
//
// WARNING: this pass is very hacky---we should not build any new functionality
// on top of this.
// -----------------------------------------------------------------------

#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"
#include "polly/Support/ISLTools.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <boost/graph/connected_components.hpp>
#include <boost/property_map/property_map.hpp>
#include <cstddef>
#include <cstdint>
#include <stdlib.h>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/detail/adjacency_list.hpp>

#include "llvm/Support/Debug.h"
#include <polly/Support/ISLOStream.h>

#define DEBUG_TYPE "array-partition"

#include <boost/throw_exception.hpp>

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

// DimInfo of all the dimensions (e.g., DimInfo only has knowledge of `sub1` of
// the dimensions A[sub1][sub2][sub3][sub4], DimInfoOfAllDimensions is an array
// of all four dimensions).
using DimInfoOfAllDimensions = std::vector<DimInfo>;

namespace {

struct PartitionInfo {
  unsigned dimension;
  unsigned factor;
  std::string style;
};

struct AccessInfo {
  std::map<Instruction *, isl::set> accessMaps;

  std::map<Instruction *, unsigned> instToScopId;

  // Base to the set of instructions storing to this base
  // NOTE: llvm::SetVector here  since it preserves the order in which
  // instructions are inserted as well as functioning like a set
  std::map<Value *, llvm::SetVector<Instruction *>> baseToInsts;

  bool sameScop(Instruction *i, Instruction *j) const {
    if (!instToScopId.count(i))
      return false;

    if (!instToScopId.count(j))
      return false;

    return instToScopId.count(i) == instToScopId.count(j);
  }
};

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

// Helper function for getting StringLiterals from LLVM value
std::optional<StringRef> extractStringLiteral(Value *v) {
  auto *gv = dyn_cast<GlobalVariable>(v->stripPointerCasts());
  if (!gv || !gv->hasInitializer())
    return std::nullopt;

  auto *cda = dyn_cast<ConstantDataArray>(gv->getInitializer());
  if (!cda || !cda->isCString())
    return std::nullopt;

  return cda->getAsCString();
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
                       Type *newArrayType, const DimInfoOfAllDimensions &info) {

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
    LLVM_DEBUG(gepInst->dump(););
    llvm::report_fatal_error(
        "Trying to change the operands of an operation with unhandled type.");
  }

  auto *gep = dyn_cast<GetElementPtrInst>(gepInst);
  // Sanity check: Avoid chain of GEPs:
  if (isa<GetElementPtrInst>(gep->getPointerOperand())) {
    llvm::report_fatal_error("Chain of GEPs is unsupported!\n");
  }

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

/// \note: It returns DimInfoOfAllDimensions info. info[0] gives the information
/// of the outer-most dimension (same convention as GEP: from outer to inner)
DimInfoOfAllDimensions extractDimInfo(const isl::set &range,
                                      llvm::Type *allocaElemType) {

  LLVM_DEBUG(llvm::errs() << "Extracting dimension info for access range "
                          << range << "\n";);

  DimInfoOfAllDimensions info;

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

    if (diffs.size() == 0) {
      assert(reachableIndices.size() == 1);
      info.emplace_back(reachableIndices.front(), 1, reachableIndices.size());
    } else if (diffs.size() != 1) {
      LLVM_DEBUG(llvm::errs()
                     << "Dim " << i << " doesn't a single step!\nIndices:\n";
                 for (auto idx : reachableIndices) {
                   llvm::errs() << "Index" << idx << "\n";
                 });
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
                              const DimInfoOfAllDimensions &dims) {
  Type *elemTy = baseElementType;
  // Reverse because dim is from outer to inner, but here the construction is
  // from inner to outer
  for (auto [init, step, elems] : llvm::reverse(dims)) {
    elemTy = ArrayType::get(elemTy, elems);
  }
  return elemTy;
}

AllocaInst *createAlloca(AllocaInst *origAlloca,
                         const DimInfoOfAllDimensions &info) {
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

      // If both memory accesses are loads: they will not overwrite each others
      // results so they can be placed in two separate banks (but if a store has
      // an overlapping index with both of them, then all 3 instructions must be
      // placed in the same bank).
      if (isa<LoadInst>(inst1) && isa<LoadInst>(inst2))
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
    DimInfoOfAllDimensions &info, const std::vector<unsigned> &indices,
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

    LLVM_DEBUG(llvm::errs() << "Range: " << range << "\n";);

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

// Function that returns map of arrayNames -> partitionInfo for later partition
// has side effect of removing all call sites of pragma markers and deleting the
// pragma marker function
//
// NOTE: This could be made more general, i.e. providing
// the name of the pragma marker function and parsing out the function arguemnts
// and naming them later/providing a handler function that maps from argument
// index to struct field. Overkill if this is the only occurance for this
std::map<std::string, PartitionInfo> collectAndErasePragmaMarkers(Function &f) {
  std::map<std::string, PartitionInfo> result;
  std::vector<CallInst *> callSites;
  Function *markerFn = nullptr;

  for (auto &bb : f) {
    for (auto &inst : bb) {
      auto *call = dyn_cast<CallInst>(&inst);
      if (!call) {
        continue;
      }

      Function *callee = call->getCalledFunction();
      if (!callee || callee->getName() != "__dyn_array_partition") {
        continue;
      }

      markerFn = callee;

      if (call->arg_size() != 4) {
        llvm::report_fatal_error(
            Twine("__dyn_array_partition: expected 4 arguments, got ") +
            Twine(call->arg_size()));
      }

      auto arrName = extractStringLiteral(call->getArgOperand(0));
      auto *dimConst = dyn_cast<ConstantInt>(call->getArgOperand(1));
      auto *factorConst = dyn_cast<ConstantInt>(call->getArgOperand(2));
      auto style = extractStringLiteral(call->getArgOperand(3));

      if (!arrName)
        llvm::report_fatal_error(
            "__dyn_array_partition: could not recover array name "
            "string literal");
      if (!dimConst || !factorConst)
        llvm::report_fatal_error(
            "__dyn_array_partition: dimension/factor must be "
            "constant integers");
      if (!style)
        llvm::report_fatal_error(
            "__dyn_array_partition: could not recover style string"
            "literal");

      LLVM_DEBUG(llvm::errs()
                 << "Partitioning: " << arrName << "\n\t" << dimConst << "\n\t"
                 << factorConst << "\n\t" << style << "\n");

      result[arrName->str()] = PartitionInfo{
          static_cast<unsigned>(dimConst->getZExtValue()),
          static_cast<unsigned>(factorConst->getZExtValue()), style->str()};

      callSites.push_back(call);
    }
  }

  // Finally remove all found call sites from the function
  for (auto *call : callSites)
    call->eraseFromParent();

  // Finally remove the external function declaration
  if (markerFn && markerFn->use_empty())
    markerFn->eraseFromParent();

  return result;
}

void rewriteAccessWithBranching(Instruction *inst, AllocaInst *baseAlloca,
                                ArrayRef<AllocaInst *> banks,
                                ArrayRef<DimInfo> bankInfo,
                                const PartitionInfo &partInfo,
                                unsigned accessIdx) {

  StringRef arrName = baseAlloca->getName();

  Instruction *baseGEPOrInst = findBaseGEP(inst);
  auto *gepInst = dyn_cast<GetElementPtrInst>(baseGEPOrInst);

  auto *load = dyn_cast<LoadInst>(inst);
  auto *store = dyn_cast<StoreInst>(inst);
  StringRef opKind = load ? "load" : "store";

  if (!gepInst) {
    if (baseGEPOrInst != inst) {
      llvm::report_fatal_error(
          "__dyn_array_partition: expected a GEP for this access");
    }

    // Early return in case we have a 0 access in to an array. i.e. there is no
    // gep construction. In this case we also do not need to change the access
    // pattern since the 0th element is always in the 0th bank
    if (load)
      load->setOperand(0, banks[0]);
    else if (store)
      store->setOperand(1, banks[0]);
    return;
  }

  // If the wanted dimension to partition from exceeds the available indices in
  // the gep instruction we can not change the indices accordingly
  if (partInfo.dimension >= gepInst->getNumIndices()) {
    llvm::report_fatal_error("__dyn_array_partition: dimension exceeds the "
                             "number of indices in this access");
  }

  // For gep instruciton:
  // %p = getelementptr [10 x [10 x i32]], ptr %arr, i64 0, i64 1, 5
  // and dimension = 2 we'd want origIdx to be value 5 in this example
  Value *origIdx = *(gepInst->idx_begin() + partInfo.dimension);
  Type *addrTy = origIdx->getType();

  unsigned factor = static_cast<unsigned>(banks.size());
  unsigned totalSize = 0;
  for (auto &[firstIndex, step, elems] : bankInfo)
    totalSize += elems;

  unsigned chunkSize = totalSize / factor;
  unsigned remainder = totalSize % factor;

  LLVMContext &ctx = inst->getContext();
  Function *f = inst->getParent()->getParent();

  // Since LLVM might reuse gep instructions we have to insert duplicate ones to
  // ensure that the later split can happen correctly. Otherwise we might split
  // well before the current instruction that we'd like to split on
  if (gepInst->getNextNode() != inst) {
    gepInst = cast<GetElementPtrInst>(gepInst->clone());
    gepInst->insertBefore(inst);
  }

  // Splits origBB right before the GEP: everything from the GEP onward
  // moves into mergeBB.
  //
  // origBB:
  //  ...
  //  %gep = getelementptr ...;
  //  %v = load ...;
  //  br %next
  //
  // ->
  //
  // origBB:
  //  ...
  //  br %mergeBB
  //
  // mergeBB:
  //  %gep = getelementptr ...;
  //  %v = load ...;
  //  br %next
  BasicBlock *origBB = gepInst->getParent();
  BasicBlock *mergeBB =
      SplitBlock(origBB, gepInst->getIterator(), (DominatorTree *)nullptr,
                 (LoopInfo *)nullptr, (MemorySSAUpdater *)nullptr,
                 "partition." + arrName + ".merge" + Twine(accessIdx));
  Instruction *placeholderBr = origBB->getTerminator();

  IRBuilder<> preBuilder(placeholderBr);
  Type *i64Ty = Type::getInt64Ty(ctx);

  // Computation of th bank that the index falls into
  Value *bankIdxNative;
  if (partInfo.style == "cyclic") {
    bankIdxNative = preBuilder.CreateURem(
        origIdx, ConstantInt::get(addrTy, factor), "bank.idx");
  } else { // "block" or "complete"
    Value *raw = preBuilder.CreateUDiv(
        origIdx, ConstantInt::get(addrTy, chunkSize), "bank.raw");
    if (remainder != 0) {
      Value *maxBank = ConstantInt::get(addrTy, factor - 1);
      Value *tooLarge = preBuilder.CreateICmpUGT(raw, maxBank);
      bankIdxNative =
          preBuilder.CreateSelect(tooLarge, maxBank, raw, "bank.idx");
    } else {
      bankIdxNative = raw;
    }
  }

  // Normalize to i64 for the comparison chain below, regardless of addrTy.
  // NOTE: I had issues with i32 vs i64 indices before
  Value *bankIdx =
      bankIdxNative->getType() == i64Ty
          ? bankIdxNative
          : preBuilder.CreateZExtOrTrunc(bankIdxNative, i64Ty, "bank.idx.64");

  placeholderBr->eraseFromParent();

  // One basic block per bank for either load/store
  std::vector<BasicBlock *> bankBBs;
  bankBBs.reserve(factor);
  for (unsigned bank = 0; bank < factor; ++bank)
    bankBBs.push_back(BasicBlock::Create(
        ctx, "partition." + arrName + "." + opKind + "." + Twine(bank), f));

  if (factor == 1) {
    IRBuilder<>(origBB).CreateBr(bankBBs[0]);
  } else {
    // if/else-if chain:
    //   bankIdx == 0 ? bank0 : (bankIdx == 1 ? bank1 : ... : bank[factor-1])
    BasicBlock *currentBB = origBB;
    for (unsigned b = 0; b + 1 < factor; ++b) {
      IRBuilder<> checkBuilder(currentBB);
      Value *cmp = checkBuilder.CreateICmpEQ(
          bankIdx, ConstantInt::get(i64Ty, b), "bank.cmp." + Twine(b));
      bool isLastCheck = (b + 2 == factor);
      BasicBlock *elseBB =
          isLastCheck
              ? bankBBs[factor - 1]
              : BasicBlock::Create(
                    ctx, "partition." + arrName + ".cmp." + Twine(b + 1), f);
      checkBuilder.CreateCondBr(cmp, bankBBs[b], elseBB);
      currentBB = elseBB;
    }
  }

  // Body for the load/store basic blocks
  std::vector<std::pair<BasicBlock *, Value *>> incoming;
  for (unsigned b = 0; b < factor; ++b) {
    IRBuilder<> caseBuilder(bankBBs[b]);
    auto [firstIndex, step, elems] = bankInfo[b];

    // Creation of bank specific target index
    //   newTargetIdx = (origIdx - firstIndex) / step
    // e.g. firstIndex=50, step=1 (block bank covering global 50..99):
    //   %sub = sub i64 %origIdx, 50
    //   %part.idx = udiv i64 %sub, 1        ; a[73] -> bank[23]
    Value *newTargetIdx = caseBuilder.CreateUDiv(
        caseBuilder.CreateSub(origIdx, ConstantInt::get(addrTy, firstIndex)),
        ConstantInt::get(addrTy, step), "part.idx");

    std::vector<Value *> newIndices;
    newIndices.reserve(gepInst->getNumIndices());

    // GEP indices are either the same as before or transformed exactly in the
    // case where they match the target dimension
    for (unsigned i = 0; i < gepInst->getNumIndices(); ++i) {
      auto *newIndex =
          i == partInfo.dimension ? newTargetIdx : *(gepInst->idx_begin() + i);
      newIndices.push_back(newIndex);
    }

    Value *newGEP = caseBuilder.CreateInBoundsGEP(
        banks[b]->getAllocatedType(), banks[b], newIndices, "part.gep");

    if (load) {
      LoadInst *newLoad = caseBuilder.CreateLoad(load->getType(), newGEP,
                                                 load->getName() + ".part");
      newLoad->setAlignment(load->getAlign());
      incoming.emplace_back(bankBBs[b], newLoad);
    } else if (store) {
      StoreInst *newStore =
          caseBuilder.CreateStore(store->getValueOperand(), newGEP);
      newStore->setAlignment(store->getAlign());
    }
    caseBuilder.CreateBr(mergeBB);
  }

  // In case we were working with a load we need to unify using phi nodes
  if (load) {
    IRBuilder<> mergeBuilder(&mergeBB->front());
    PHINode *phi = mergeBuilder.CreatePHI(load->getType(), factor,
                                          load->getName() + ".merged");
    for (auto &[bb, val] : incoming)
      phi->addIncoming(val, bb);
    load->replaceAllUsesWith(phi);
  }

  inst->eraseFromParent();
  // Only conditionally delete the gep instruction. LLVM may reuse previous gep
  // instructions to access different parts of the array, as such we should only
  // delete once there is no use left for the calculated gep
  if (gepInst->use_empty()) {
    gepInst->eraseFromParent();
  }
}

// Helper function for gathering type of inner arrays in multidimensional arrays
ArrayType *getTargetDimType(Type *ty, unsigned dimension) {
  for (unsigned i = 1; i < dimension; ++i) {
    auto *at = dyn_cast<ArrayType>(ty);
    if (!at)
      llvm::report_fatal_error(
          "__dyn_array_partition: dimension exceeds array rank");
    ty = at->getElementType();
  }
  auto *at = dyn_cast<ArrayType>(ty);
  if (!at)
    llvm::report_fatal_error(
        "__dyn_array_partition: dimension exceeds array rank");
  return at;
}

// Recursive funtion for rebuilding the array type. Go through each dimension
// and generate the corresponding array type
Type *getPartitionedArrayType(Type *ty, unsigned dimension, unsigned numElems) {
  auto *at = cast<ArrayType>(ty);
  if (dimension == 1)
    return ArrayType::get(at->getElementType(), numElems);
  return ArrayType::get(
      getPartitionedArrayType(at->getElementType(), dimension - 1, numElems),
      at->getNumElements());
}

std::tuple<std::vector<AllocaInst *>, std::vector<DimInfo>>
createPartitionBankAllocas(AllocaInst *baseAlloca, const PartitionInfo &info) {
  // Get total size of the dimension we'd like to partition accross
  unsigned totalSize =
      getTargetDimType(baseAlloca->getAllocatedType(), info.dimension)
          ->getArrayNumElements();

  if (totalSize < 1) {
    llvm::report_fatal_error(
        "__dyn_array_partition: cannot partition a less than 1 length array");
  }

  unsigned factor = info.style == "complete" ? totalSize : info.factor;

  if (factor == 0 || factor > totalSize) {
    llvm::report_fatal_error("__dyn_array_partition: factor must be in [1, N]");
  }

  // Logic for determining bank partition numbers
  unsigned chunkSize = totalSize / factor;
  unsigned remainder = totalSize % factor;

  std::vector<DimInfo> banks;
  banks.reserve(factor);
  if (info.style == "block" || info.style == "complete") {
    unsigned offset = 0;
    for (unsigned bank = 0; bank < factor; ++bank) {
      // Put all remaining elements into the last bank
      unsigned elems = chunkSize + (bank + 1 == factor ? remainder : 0);
      banks.emplace_back(offset, 1, elems);
      offset += elems;
    }
  } else if (info.style == "cyclic") {
    for (unsigned bank = 0; bank < factor; ++bank) {
      unsigned elems = chunkSize + (bank < remainder ? 1u : 0u);
      banks.emplace_back(bank, factor, elems);
    }
  } else {
    llvm::report_fatal_error(Twine("__dyn_array_partition: unknown style '") +
                             info.style +
                             "' (expected block, cyclic, or complete)");
  }

  // Insertion of allocations
  std::vector<AllocaInst *> bankAllocas;
  bankAllocas.reserve(banks.size());

  IRBuilder<> builder(baseAlloca->getNextNode());
  for (auto &[firstIndex, step, elems] : banks) {
    Type *bankType = getPartitionedArrayType(baseAlloca->getAllocatedType(),
                                             info.dimension, elems);
    AllocaInst *newAlloca = builder.CreateAlloca(
        bankType, baseAlloca->getArraySize(), baseAlloca->getName());
    newAlloca->setAlignment(baseAlloca->getAlign());
    bankAllocas.push_back(newAlloca);
  }

  return {bankAllocas, banks};
}

} // namespace

struct ArrayPartition : PassInfoMixin<ArrayPartition> {
  PreservedAnalyses run(Function &f, FunctionAnalysisManager &fam);
};

PreservedAnalyses ArrayPartition::run(Function &f,
                                      FunctionAnalysisManager &fam) {

  if (f.getName() == "main") {
    LLVM_DEBUG(llvm::errs()
               << "Skipping main function for automatic array partitioning!\n");
    return PreservedAnalyses::all();
  }

  auto pragmaInfo = collectAndErasePragmaMarkers(f);

  AccessInfo info;

  for (auto &bb : f) {
    for (auto &inst : bb) {
      if (isa<llvm::LoadInst, llvm::StoreInst>(&inst)) {
        Value *base = findBase(&inst);
        info.baseToInsts[base].insert(&inst);
      }
    }
  }

  for (auto [base, insts] : info.baseToInsts) {
    auto it = pragmaInfo.find(base->getName().str());
    if (it != pragmaInfo.end()) {
      auto *baseAlloca = dyn_cast<AllocaInst>(base);
      if (!baseAlloca) {
        llvm::report_fatal_error(
            Twine("__dyn_array_partition: only local (alloca) arrays are "
                  "supported so far '") +
            base->getName() + "' is not one");
      }

      const PartitionInfo &partInfo = it->second;
      auto [bankAllocas, bankInfo] =
          createPartitionBankAllocas(baseAlloca, partInfo);

      unsigned accessIdx = 0;
      for (Instruction *inst : insts) {
        rewriteAccessWithBranching(inst, baseAlloca, bankAllocas, bankInfo,
                                   partInfo, accessIdx);
        ++accessIdx;
      }
      continue;
    }
  }

  return PreservedAnalyses::all();
}

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
