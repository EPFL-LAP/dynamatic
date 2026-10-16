// -----------------------------------------------------------------------
// Canonicalize the GEPs computing memory addresses for the translation to
// the std dialect, which computes the element index of each GEP separately.
//
// Chains of byte-addressed GEPs (each with an i8 source element type and a
// single index) are folded into a single GEP whose offset is the sum of
// theirs. LLVM can split the address of an aligned access into byte offsets
// that are not whole numbers of elements on their own.
//
// Ex. (int values[N], accessing the element at byte offset %i - 1)
//  %p = getelementptr i8, ptr %values, i64 %i
//  %q = getelementptr i8, ptr %p, i64 -1
//  %v = load i32, ptr %q, align 4
//
// Canonicalized:
//  %sum = add i64 %i, -1
//  %q = getelementptr i8, ptr %values, i64 %sum
//  %v = load i32, ptr %q, align 4
// -----------------------------------------------------------------------

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

namespace {

struct CanonicalizeGEPsPass : public PassInfoMixin<CanonicalizeGEPsPass> {
  PreservedAnalyses run(Function &f, FunctionAnalysisManager &);
};

} // namespace

/// Returns whether the GEP adds a byte offset to its pointer.
static bool isByteGEP(GetElementPtrInst *gep) {
  return gep && gep->getSourceElementType()->isIntegerTy(8) &&
         gep->getNumIndices() == 1;
}

PreservedAnalyses CanonicalizeGEPsPass::run(Function &f,
                                            FunctionAnalysisManager &) {
  bool changed = false;
  bool folded = true;
  while (folded) {
    folded = false;
    for (BasicBlock &block : f) {
      for (Instruction &inst : make_early_inc_range(block)) {
        auto *outer = dyn_cast<GetElementPtrInst>(&inst);
        if (!isByteGEP(outer))
          continue;
        auto *inner = dyn_cast<GetElementPtrInst>(outer->getPointerOperand());
        if (!isByteGEP(inner))
          continue;
        IRBuilder<> irBuilder(outer);
        Value *innerIdx = *inner->idx_begin();
        Value *outerIdx = *outer->idx_begin();
        Type *idxType = outerIdx->getType();
        innerIdx = irBuilder.CreateSExtOrTrunc(innerIdx, idxType);
        Value *sum = irBuilder.CreateAdd(innerIdx, outerIdx);
        Value *foldedGEP = irBuilder.CreateGEP(irBuilder.getInt8Ty(),
                                               inner->getPointerOperand(), sum);
        outer->replaceAllUsesWith(foldedGEP);
        outer->eraseFromParent();
        if (inner->use_empty())
          inner->eraseFromParent();
        folded = true;
        changed = true;
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
