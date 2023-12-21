//===- NameAnalysis.h - Uniquely name all IR operations ---------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Name analysis infrastructure, allocating and giving access to unique names
// for all IR operations. This is meant to be be used from Dynamatic passes as a
// queryable analysis on the top-level module operation (`mlir::ModuleOp`) using
// `getAnalysis<NameAnalysis>()`.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_ANALYSIS_NAMEANALYSIS
#define DYNAMATIC_ANALYSIS_NAMEANALYSIS

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Pass/AnalysisManager.h"

namespace dynamatic {

/// Analysis to manage unique operation names throughout the IR's lifetime.
/// Query at the beginning of Dynamatic passes using
/// `getAnalysis<NameAnalysis>()` and cache it to avoid recomputations for
/// further passes using `markAnalysesPreserved<NameAnalysis>()` at the end.
///
/// This analysis provides an easy way to manage unique names for all IR
/// operations (besides `func` and `handshake` level functions as well as the
/// top-level MLIR module itself): performing caching, verification, providing
/// unique names to yet unnamed operations, etc. At construction time and when
/// invoking the `walk` method, the entire IR nested under the
/// constructor-provided top-level operation is walked, establishing a
/// bidirectionnal mapping between operation and their names as well as checking
/// the analysis's invariants, which are:
/// - uniqueness: no two operations can hold the same name
/// - persistence: the name of an operation cannot change once set
/// - completeness (optional, opt-in): all name-able operations are named
/// In general, it is not recommended to set operation names without going
/// through this analysis's API; doing so may break the analysis's invariants
/// and lead to undefined behavior.
///
/// Operation names are internally stored in an operation attribute of type
/// `circt::handhsake::NameAttr` attached to the operation being named under the
/// attibute name "name"; the attribute is a simple wrapper around a
/// `mlir::StringAttr`. Channel names are determined based on the names of the
/// "producing" and "consuming" operations.
class NameAnalysis {
public:
  /// Constructor called automatically by `getAnalysis<NameAnalysis>()` if the
  /// analysis is not already cached. Simply stores the top-level pass operation
  /// it is passed and walks the IR to establish mappings between already named
  /// operations and their names. Users should call the `isAnalysisValid` method
  /// after the constructor returns to verify that the analysis's invariants
  /// are not violated by the operations nested under the passed operation.
  NameAnalysis(Operation *op) : topLevelOp(op) {
    // Explictly discard the result since we can't return anything from here
    (void)walk(UnnamedBehavior::DO_NOTHING);
  };

  /// Whether the last walk through the IR revealed that one of the analysis's
  /// invariants is broken. Setting new operation names despite a broken
  /// invariant will lead to undefined behavior.
  bool isAnalysisValid() { return namesValid; }

  /// Whether the last walk through the IR found no unnamed operation it did not
  /// name. If this returns true right after the analysis walked the IR, all
  /// operations may be assumed to have names (which may not be unique, if the
  /// IR is in an invalid state).
  bool areAllOpsNamed() { return allOpsNamed; }

  /// Whether the operation has a name.
  bool hasName(Operation *op) { return opToName.contains(op); }

  /// If the operation doesn't currently have a name, sets a unique name for the
  /// operation based on its type and returns a reference to it. If the
  /// operation already had a name, just return a reference to this name.
  StringRef getName(Operation *op);

  /// Derives a unique name for the provided operand, which is based on the
  /// respective names of its "producer" and "consumer" operations as well as
  /// operation-specific result/operand names if available (if not, their
  /// respective index). The producer and/or comsumer operations are uniquely
  /// named if they do not already have names.
  std::string getName(OpOperand &oprd);

  /// Return the operation that has this name, if it exists.
  Operation *getOp(StringRef name) { return nameToOp[name]; };

  /// This is simply an alias for `getName(Operation*)` for when the
  /// programmer's intent is to just set a unique name without reading it back.
  void setName(Operation *op) { getName(op); };

  /// Attempts to name the operation with the provided name. Fails if the
  /// operation was already named or if another operation already holds this
  /// name, succeeds otherwise. If the `uniqueWhenTaken` flag is set, succeeds
  /// even if the provided name is already taken by deriving a unique name from
  /// the provided one for the operation.
  LogicalResult setName(Operation *op, StringRef name,
                        bool uniqueWhenTaken = false);

  /// Attempts to uniquely name the operation by concatenating a unique name
  /// based on the operation's type with the name of an "ascending" operation
  /// that is logically related to the operation being named. This is useful for
  /// debugging when one may want to showcase operation filiations through their
  /// name. Uniquely name the ascendant operation if it doesn't have a name yet.
  /// Fails if the operation was already named, or if another operation already
  /// holds the derived name. If the `uniqueWhenTaken` flag is set, succeeds
  /// even if the provided name is already taken by deriving a unique name from
  /// the "normal" one for the operation.
  LogicalResult setName(Operation *op, Operation *ascendant,
                        bool uniqueWhenTaken = false);

  /// Tells IR walks what to do when encountering an unnamed operation.
  enum class UnnamedBehavior {
    /// Do nothing (further calls to `areAllOpsNamed` will return false until
    /// the next walk).
    DO_NOTHING,
    /// Set a unique name for the unnamed operation.
    NAME,
    /// Report an error on standard error and make the walk fail (further calls
    /// to `areAllOpsNamed` will return false until the next walk).
    FAIL
  };

  /// Walks the IR to update known mappings between operations and their names.
  /// The argument controls the walk's behavior when encountering an unnamed
  /// operation. Fails if the IR breaks one of the analysis's invariants (in
  /// which case setting new operation names will lead to undefined behavior) or
  /// if the argument was `UnnamedBehavior::FAIL` and there exists ar least one
  /// unnamed operation in the IR.
  LogicalResult walk(UnnamedBehavior onUnnamed);

  /// Equivalent to `analysis.walk(UnnamedBehavior::NAME)` but never produces an
  /// error and asserts if some analysis invariant is broken. It is safe to call
  /// it if the analysis was just checked to be valid with `isAnalysisValid` and
  /// no operation name was set outside of the analysis since then.
  void nameAllUnnamedOps() {
    bool walkSuccess = succeeded(walk(UnnamedBehavior::NAME));
    assert(walkSuccess && "analysis invariant is broken");
  }

  /// Invalidation hook to keep the analysis cached across passes. Returns true
  /// if the analysis should be invalidated and fully reconstructed the next
  /// time it is queried.
  bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses &pa) {
    return !pa.isPreserved<NameAnalysis>();
  }

  /// Names can't be guaranteed to be unique if the analysis is copied.
  NameAnalysis(const NameAnalysis &other) = delete;
  /// Names can't be guaranteed to be unique if the analysis is copied.
  NameAnalysis operator=(const NameAnalysis &other) = delete;

private:
  /// Top-level operation on which the analysis operates. This should be the
  /// top-level MLIR module to ensure unique names across an entire module.
  Operation *topLevelOp;
  /// Operation counters, used to generate unique names on-demand based on the
  /// operation's type.
  DenseMap<mlir::OperationName, unsigned> counters;
  /// Maps all operations whose the analysis knows have names to a copy of their
  /// name. Note that an operation may still be present in the map but no longer
  /// in the IR.
  mlir::DenseMap<Operation *, std::string> opToName;
  /// Maps all known operation names to their owning operation. Note that an
  /// operation may still be present in the map but no longer in the IR. This is
  /// intentional and is meant so that the analysis never assigns a name which
  /// belonged to an erased operation to a new operation.
  llvm::StringMap<Operation *> nameToOp;
  /// Whether the last walk through the IR revealed that one of the analysis's
  /// invariants is broken. Reset to true at the beginning of each walk and
  /// potentially set to false during it.
  bool namesValid = true;
  /// Whether the last walk through the IR revealed found no unnamed operation
  /// it did not name. Reset to true at the beginning of each walk and
  /// potentially set to false during it.
  bool allOpsNamed = true;

  /// Generate a unique name based on the type of an operation (using the
  /// counters).
  std::string genUniqueName(mlir::OperationName opName);

  /// Derive a unique name from a base name by appending an increasing "counter"
  /// string at the end (e.g., "baseName0" -> "baseName1" -> ...).
  std::string deriveUniqueName(StringRef base);

  /// Adds the bidirectional mapping between an operation and its name to our
  /// maps. It's assumed that the operation/name pair was unknown before.
  void addMapping(Operation *op, StringRef name) {
    nameToOp[name] = op;
    opToName[op] = name;
  }

  /// Determines whether the operation is intrinsically named i.e., whether it
  /// has a naturally unique name that is not represented through our Handhsake
  /// attribute. If it does, updates our internal mappings to make sure it is
  /// considered taken when assigning auto-generated names to other operations.
  bool isIntrinsicallyNamed(Operation *op);

  /// When getting the "producer part" of an operand's name and it is a block
  /// argument, we can derive a meaningful name when it is a function's argument
  /// of some sort. In those cases, sets the two strings, respectively, to the
  /// function's name and the argument's name. In other cases, consider the
  /// parent operation's name and the index of the resgion/block the argument
  /// belongs to as the producer's and result name, respectively.
  void getBlockArgName(BlockArgument arg, std::string &prodName,
                       std::string &resName);
};

/// Attemps to get the unique name of the operation. Returns an empty string if
/// the operation does not have a unique name.
std::string getUniqueName(Operation *op);

/// Attemps to get the unique name of the operand, derived from its producer and
/// consumer operation respective names. Returns an empty string if the producer
/// or consumer operation does not have a name.
std::string getUniqueName(OpOperand &oprd);

} // namespace dynamatic

#endif // DYNAMATIC_ANALYSIS_NAMEANALYSIS
