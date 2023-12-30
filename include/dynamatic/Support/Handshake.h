//===- Handshake.h - Helpers for Handshake-level analysis -------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares a couple analysis functions and data-structures that
// may be helpful when working with Handshake-level IR. These may be useful in
// many different contexts and as such are not associated with any pass in
// particular.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Support/LLVM.h"

namespace dynamatic {

/// Helper class to keep memory dependencies annotations consistent when
/// lowering memory operations.
///
/// When lowering memory operations, one usually wants to replace a memory
/// access with an equivalent one from a "lower-level" dialect while keeping the
/// memory-related attributes (interface and dependence information)
/// semantically equivalent. However, each `handshake::MemoryDependenceAttr`
/// attribute refers to the memory access with which the dependency exists via
/// its unique name, which becomes incorrect when the referenced access changes
/// name after its associated operation is lowered. The class provides an easy
/// way to record such memory access replacements and rename any reference to
/// them in memory dependence annotations across the IR.
class MemoryOpLowering {
public:
  /// Constructs an instance of the class from a reference to a naming analysis
  /// that encompasses all memory accesses that are going to be replaced.
  MemoryOpLowering(NameAnalysis &namer) : namer(namer){};

  /// Records a replacement from the old operation to the new operation (both
  /// are meant to be memory accesses), naming both in the process if they were
  /// not previously named. In addition, copies the
  /// `handshake::MemoryDependenceArrayAttr` attribute (and the
  /// `handshake::MemoryInterfaceAttr` attribute, if the flag is set) from the
  /// old access to the new one, if present.
  void recordReplacement(Operation *oldOp, Operation *newOp,
                         bool forwardInterface = true);

  /// Walks the IR under the given operation looking for all memory accesses. If
  /// a memory access is annotated with dependencies to other accesses and if
  /// some of these accesses have been replaced (as recorded by a call to
  /// `MemoryOpLowering::recordReplacement`), rewrites them to reference the new
  /// operation's name instead of the old one. Returns true when at least one
  /// memory dependence was modified due to memory access renaming.
  bool renameDependencies(Operation *topLevelOp);

private:
  /// Reference to the naming analysis, used to set/get memory operation's name
  /// when calling `recordReplacement`.
  NameAnalysis &namer;
  /// Records memory operation replacements, mapping the name of the operation
  /// being replaced to the name of the operation it is replaced with.
  llvm::StringMap<std::string> nameChanges;
};

/// Determines whether the given value has any "real" use i.e., a use which is
/// not the operand of a sink. If this function returns false for a given value
/// and one decides to erase the operation that defines it, one should keep in
/// mind that there may still be actual users of the value in the IR. In this
/// situation, using `eraseSinkUsers` in conjunction with this function will get
/// rid of all of the value's users.
bool hasRealUses(Value val);

/// Erases all sink operations that have the given value as operand.
void eraseSinkUsers(Value val);

/// Erases all sink operations that have the given value as operand. Uses the
/// rewriter to erase operations.
void eraseSinkUsers(Value val, PatternRewriter &rewriter);

/// Identifies the subset of the control operation's results that are part of
/// the control path to the LSQ interface. The control operations' results that
/// are not of type `NoneType` are ignored and will never be part of the
/// returned vector. Typically, one would call this function on a (lazy-)fork
/// directly providing a group allocation signal to the LSQ to inquire about
/// other fork results that would trigger other group allocations. The returned
/// values are guaranteed to be in the same order as the control operation's
/// results.
SmallVector<Value> getLSQControlPaths(circt::handshake::LSQOp lsqOp,
                                      Operation *ctrlOp);

} // namespace dynamatic