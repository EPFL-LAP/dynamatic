// -----------------------------------------------------------------------
// Canonicalize the GEPs computing memory addresses for the translation to
// the std dialect, which flattens all arrays into 1D memrefs.
//
// Each chain of GEPs computing an address used by a non-GEP instruction (e.g.,
// a load or a store) is collapsed into a single GEP with a single index: the
// flattened offset from the base pointer. The offset counts the largest unit
// that divides the size of the chain's elements and the offsets of all its
// indices (the elements themselves, unless the chain adds byte offsets that are
// not whole elements).
//
// Ex. 1 (int mat[N][M], accessing mat[i][j])
//  %row = getelementptr [N x [M x i32]], ptr %mat, i64 0, i64 %i
//  %elem = getelementptr [M x i32], ptr %row, i64 0, i64 %j
//  %v = load i32, ptr %elem, align 4
//
// Canonicalized:
//  %mul = mul i64 %i, M
//  %idx = add i64 %mul, %j
//  %elem = getelementptr i32, ptr %mat, i64 %idx
//  %v = load i32, ptr %elem, align 4
//
// Ex. 2 (int values[N], accessing the element at byte offset %i - 1)
//  %p = getelementptr i8, ptr %values, i64 %i
//  %q = getelementptr i8, ptr %p, i64 -1
//  %v = load i32, ptr %q, align 4
//
// Canonicalized:
//  %idx = add i64 %i, -1
//  %q = getelementptr i8, ptr %values, i64 %idx
//  %v = load i32, ptr %q, align 4
// -----------------------------------------------------------------------

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include <cstdlib>
#include <numeric>

using namespace llvm;

namespace {

struct CanonicalizeGEPsPass
    : public OptionalPassInfoMixin<CanonicalizeGEPsPass> {
  PreservedAnalyses run(Function &f, FunctionAnalysisManager &);
};

/// An index of a GEP and the number of bytes the address advances when the
/// index increases by one.
struct OffsetTerm {
  Value *index;
  uint64_t stride;
};

} // namespace

/// Returns the innermost element type of a (possibly nested) array type.
static Type *getScalarType(Type *type) {
  while (type->isArrayTy())
    type = type->getArrayElementType();
  return type;
}

/// Returns the base pointer of the chain of GEPs ending at `gep` and the offset
/// terms of the chain, from the base pointer to `gep`. The base pointer is
/// nullptr if a GEP of the chain indexes into a non-array type (e.g., a
/// struct).
static std::pair<Value *, SmallVector<OffsetTerm>>
getBaseAndOffsetTerms(GetElementPtrInst *gep, const DataLayout &dl) {
  SmallVector<OffsetTerm> terms;
  Value *ptr = gep;
  while (auto *chainGEP = dyn_cast<GetElementPtrInst>(ptr)) {
    SmallVector<OffsetTerm> gepTerms;
    // The first index steps over the source element type, the next ones over
    // the elements of the arrays nested in it
    Type *type = chainGEP->getSourceElementType();
    for (auto [i, index] : llvm::enumerate(chainGEP->indices())) {
      if (i > 0) {
        // The index selects a struct field (e.g., "getelementptr {i32, [4 x
        // i16]}, ptr %s, i64 0, i32 1") or a vector element. The offset of a
        // struct field is not a multiple of the index, so the chain cannot be
        // collapsed into a stride-based offset
        if (!type->isArrayTy())
          return {nullptr, {}};
        type = type->getArrayElementType();
      }
      gepTerms.push_back(
          {/*index=*/index,
           /*stride=*/dl.getTypeAllocSize(type).getFixedValue()});
    }
    terms.insert(terms.begin(), gepTerms.begin(), gepTerms.end());
    ptr = chainGEP->getPointerOperand();
  }
  return {ptr, terms};
}

/// Returns the larger of the two types in memory, or the other one if one of
/// them is nullptr.
static Type *getLargerType(Type *lhs, Type *rhs, const DataLayout &dl) {
  if (!lhs || !rhs)
    return lhs ? lhs : rhs;
  return dl.getTypeAllocSize(rhs).getFixedValue() >
                 dl.getTypeAllocSize(lhs).getFixedValue()
             ? rhs
             : lhs;
}

/// Returns the largest of the element types of the chain of GEPs ending at
/// `gep`.
static Type *getLargestGEPScalarType(GetElementPtrInst *gep,
                                     const DataLayout &dl) {
  Type *scalarType = nullptr;
  for (auto *chainGEP = gep; chainGEP;
       chainGEP = dyn_cast<GetElementPtrInst>(chainGEP->getPointerOperand()))
    scalarType = getLargerType(
        scalarType, getScalarType(chainGEP->getSourceElementType()), dl);
  return scalarType;
}

/// Returns the largest of the types that the users of `gep` load or store
/// through it, or nullptr if no user is a load or a store. The accessed types
/// matter when the GEPs only have byte element types: e.g., newer clang
/// versions access the elements of "int a[N][32]" (i.e., i32) through
/// "getelementptr [128 x i8]" and "getelementptr [4 x i8]", and load them as
/// i32.
static Type *getLargestAccessedScalarType(GetElementPtrInst *gep,
                                          const DataLayout &dl) {
  Type *scalarType = nullptr;
  for (User *user : gep->users())
    if (isa<LoadInst, StoreInst>(user) &&
        getLoadStorePointerOperand(user) == gep)
      scalarType =
          getLargerType(scalarType, getScalarType(getLoadStoreType(user)), dl);
  return scalarType;
}

/// Sums the values with a balanced tree of additions.
static Value *buildAdderTree(IRBuilder<> &irBuilder, ArrayRef<Value *> values) {
  if (values.size() == 1)
    return values.front();
  size_t mid = values.size() / 2;
  Value *lhs = buildAdderTree(irBuilder, values.take_front(mid));
  Value *rhs = buildAdderTree(irBuilder, values.drop_front(mid));
  return irBuilder.CreateAdd(lhs, rhs);
}

/// Collapses the chain of GEPs ending at `gep` into a single GEP with a single
/// index. Returns the new GEP, or nullptr if the chain is already canonical or
/// cannot be collapsed.
static GetElementPtrInst *collapseGEPChain(GetElementPtrInst *gep,
                                           const DataLayout &dl) {
  auto [base, terms] = getBaseAndOffsetTerms(gep, dl);
  if (!base)
    return nullptr;
  // NOTE: This does not guarantee that the type has the same bitwidth as the
  // array's elements in the original C code: e.g., clang may access a
  // _BitInt(5) array through i8 GEPs and i8 loads, or an int array through
  // byte GEPs whose users are not loads or stores (e.g., selects). The
  // translator compares sizes in memory against the memref's element type
  Type *scalarType = getLargerType(getLargestGEPScalarType(gep, dl),
                                   getLargestAccessedScalarType(gep, dl), dl);
  assert(scalarType && "chain of GEPs without element type");
  uint64_t scalarSize = dl.getTypeAllocSize(scalarType).getFixedValue();
  if (scalarSize == 0)
    return nullptr;

  // The constant indices add up to a single byte offset
  int64_t constOffset = 0;
  SmallVector<OffsetTerm> varTerms;
  for (OffsetTerm &term : terms) {
    if (auto *constIndex = dyn_cast<ConstantInt>(term.index))
      constOffset += constIndex->getSExtValue() * (int64_t)term.stride;
    else
      varTerms.push_back(term);
  }

  // The offset counts the largest unit that divides the element size and the
  // offsets of all indices
  uint64_t unit = std::gcd(scalarSize, (uint64_t)std::abs(constOffset));
  for (OffsetTerm &term : varTerms)
    unit = std::gcd(unit, term.stride);
  Type *unitType = unit == scalarSize
                       ? scalarType
                       : IntegerType::get(gep->getContext(), unit * 8);

  if (gep->getNumIndices() == 1 &&
      !isa<GetElementPtrInst>(gep->getPointerOperand()) &&
      gep->getSourceElementType() == unitType)
    return nullptr;

  IRBuilder<> irBuilder(gep);
  Type *indexType = dl.getIndexType(gep->getPointerOperandType());
  SmallVector<Value *> scaledIndices;
  for (OffsetTerm &term : varTerms) {
    Value *index = irBuilder.CreateSExtOrTrunc(term.index, indexType);
    if (uint64_t factor = term.stride / unit; factor != 1)
      index = irBuilder.CreateMul(index, ConstantInt::get(indexType, factor));
    scaledIndices.push_back(index);
  }
  if (constOffset != 0 || scaledIndices.empty())
    scaledIndices.push_back(ConstantInt::get(
        indexType, constOffset / (int64_t)unit, /*IsSigned=*/true));
  Value *offset = buildAdderTree(irBuilder, scaledIndices);

  // Insert the instruction directly: the builder would fold a GEP of a global
  // with a constant offset into a constant expression
  auto *collapsedGEP =
      irBuilder.Insert(GetElementPtrInst::Create(unitType, base, {offset}));
  collapsedGEP->takeName(gep);
  return collapsedGEP;
}

PreservedAnalyses CanonicalizeGEPsPass::run(Function &f,
                                            FunctionAnalysisManager &) {
  const DataLayout &dl = f.getParent()->getDataLayout();

  // Collapse each chain ending at a GEP with a non-GEP user. GEPs that only
  // feed other GEPs are collapsed as part of the chains they belong to
  SmallVector<GetElementPtrInst *> chainEnds;
  for (Instruction &inst : instructions(f))
    if (auto *gep = dyn_cast<GetElementPtrInst>(&inst))
      if (llvm::any_of(gep->users(), [](User *user) {
            return !isa<GetElementPtrInst>(user);
          }))
        chainEnds.push_back(gep);

  bool changed = false;
  for (GetElementPtrInst *gep : chainEnds) {
    GetElementPtrInst *collapsedGEP = collapseGEPChain(gep, dl);
    if (collapsedGEP) {
      gep->replaceUsesWithIf(collapsedGEP, [](Use &use) {
        return !isa<GetElementPtrInst>(use.getUser());
      });
      changed = true;
    }
  }

  // Erase the GEPs that the collapsed ones replaced
  bool erased = true;
  while (erased) {
    erased = false;
    for (Instruction &inst : make_early_inc_range(instructions(f))) {
      if (isa<GetElementPtrInst>(inst) && inst.use_empty()) {
        inst.eraseFromParent();
        erased = true;
      }
    }
  }
  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "CanonicalizeGEPs", LLVM_VERSION_STRING,
          [](PassBuilder &pb) {
            pb.registerPipelineParsingCallback(
                [](StringRef name, FunctionPassManager &fpm,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (name == "canonicalize-geps") {
                    fpm.addPass(CanonicalizeGEPsPass());
                    return true;
                  }
                  return false;
                });
          }};
}
