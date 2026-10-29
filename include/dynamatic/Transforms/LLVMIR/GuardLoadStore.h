#ifndef DYNAMATIC_TRANSFORMS_LLVMIR_GUARDLOADSTORE_H
#define DYNAMATIC_TRANSFORMS_LLVMIR_GUARDLOADSTORE_H

#include "llvm/IR/PassManager.h"

namespace dynamatic {

struct GuardLoadStorePass
    : public llvm::OptionalPassInfoMixin<GuardLoadStorePass> {
  llvm::PreservedAnalyses run(llvm::Function &f,
                              llvm::FunctionAnalysisManager &);
};

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_LLVMIR_GUARDLOADSTORE_H
