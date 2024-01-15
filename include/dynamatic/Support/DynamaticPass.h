//===- DynamaticPass.h - Base class for Dynamatic passes --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Support for pass writing in Dynamatic. Defines the `DynamaticPass` class
// which TableGen-generated pass classes (e.g.,
// `dynamatic::impl::<PassName>Base`) may choose to inherit from to get access
// to classic invariant checking logic and avoid common boilerplate code.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_DYNAMATICPASS_H
#define DYNAMATIC_SUPPORT_DYNAMATICPASS_H

#include "dynamatic/Analysis/NameAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <functional>

namespace dynamatic {

/// Abstract base class for Dynamatic passes, performing domain-specific
/// pass pre/post-conditions verification on demand around a user-implemented
/// pass hook that is called on the top-level `mlir::ModuleOp` operation. This
/// pass is meant to factor in common invariant checks and analysis steps that
/// most Dynamatic passes care about (e.g., naming analysis).
///
/// Implementors should override the `runDynamaticPass` method instead of the
/// standard `runOnOperation` method to implement their pass's logic; the latter
/// is defined by this class and performs invariant checking around the call to
/// `runDynamaticPass`.
class DynamaticPass : public mlir::OperationPass<mlir::ModuleOp> {
protected:
  /// Simply forwards the pass's type ID to the parent `mlir::OperationPass`.
  DynamaticPass(mlir::TypeID passID)
      : mlir::OperationPass<mlir::ModuleOp>(passID) {}
  /// Default copy constructor.
  DynamaticPass(const DynamaticPass &other) = default;

  /// Hook for the pass called for the top-level MLIR module. Performs invariant
  /// checking before and after running the actual user-provided pass logic.
  void runOnOperation() override {
    // Make sure all operation names are unique and haven't changed from what is
    // cached (if anything is cached)
    NameAnalysis &nameAnalysis = getAnalysis<NameAnalysis>();
    if (!nameAnalysis.isAnalysisValid())
      return signalPassFailure();

    // Run the actual pass
    runDynamaticPass();

    // Make sure all operation names are unique and haven't changed from what is
    // cached. Also name operations that do not currently have a name (unless
    // instructed otherwise)
    auto onUnnamed = dontNameOps ? NameAnalysis::UnnamedBehavior::DO_NOTHING
                                 : NameAnalysis::UnnamedBehavior::NAME;
    if (failed(nameAnalysis.walk(onUnnamed)))
      return signalPassFailure();

    // The name analysis is always preserved across passes
    markAnalysesPreserved<NameAnalysis>();
  }

  /// Entry point for Dynamatic passes that replaces the normal `runOnOperation`
  /// hook. It is called from `DynamaticPass::runOnOperation` after checking for
  /// some pass pre-conditions. After it returns, pass post-conditions are
  /// checked before concluding the pass.
  virtual void runDynamaticPass() = 0;

  /// After `runDynamaticPass` returns, do not create unique names for
  /// operations that currently have no name.
  void doNotNameOperations() { dontNameOps = true; }

private:
  /// Whether to name unnamed operations after the pass.
  bool dontNameOps = false;
};

} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_DYNAMATICPASS_H
