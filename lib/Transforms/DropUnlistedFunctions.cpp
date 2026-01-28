#include "dynamatic/Transforms/DropUnlistedFunctions.h"

#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace dynamatic;

// Boilerplate: Include this for the pass option defintitions
namespace dynamatic {
// import auto-generated base class definition "DropUnlistedFunctionsBase" and
// put it under the dynamatic namespace.
#define GEN_PASS_DEF_DROPUNLISTEDFUNCTIONS
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace dynamatic

namespace {
struct DropUnlistedFunctionsPass
    : public dynamatic::impl::DropUnlistedFunctionsBase<
          DropUnlistedFunctionsPass> {

  /// \note: Use the auto-generated construtors from tblgen
  using DropUnlistedFunctionsBase::DropUnlistedFunctionsBase;
  void runOnOperation() override;
};

void DropUnlistedFunctionsPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ModuleOp modOp = llvm::dyn_cast<ModuleOp>(getOperation());
  OpBuilder builder(ctx);

  llvm::SmallDenseSet<StringRef> keepSet(function_names.begin(),
                                         function_names.end());

  // note:
  // - llvm::make_early_inc_range: safety iterate through an iterator when there
  // might be changes to them.
  for (auto func :
       llvm::make_early_inc_range(modOp.getOps<LLVM::LLVMFuncOp>())) {
    assert(func->use_empty());
    if (!keepSet.contains(func.getName())) {
      func->erase();
    }
  }
}

} // namespace
