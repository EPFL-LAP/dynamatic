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

#ifndef DYNAMATIC_SUPPORT_HANDSHAKE_H
#define DYNAMATIC_SUPPORT_HANDSHAKE_H

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Support/CFG.h"
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

/// Helper class to instantiate appropriate Handshake-level memory interfaces
/// (handshake::MemoryControllerOp and/or handshake::LSQOp) for a set of memory
/// accesses. This abstracts away the complexity of determining the kind of
/// memory interface(s) one needs for a set of memory accesses, the somewhat
/// convoluted creation of SSA inputs for these interface(s), and the "circuit
/// rewiring" requires to connect accesses to interfaces.
///
/// Add memory ports (i.e., load/store-like operations) to the future memory
/// interfaces using `MemoryInterfaceBuilder::addMCPort` and
/// `MemoryInterfaceBuilder::addLSQPort`, then instantiate the interfaces using
/// `MemoryInterfaceBuilder::instantiateInterfaces`. The memory ports' addition
/// order to the builder are reflected in the input ordering of instantiated
/// interfaces.
class MemoryInterfaceBuilder {
public:
  /// Constructs the memory interface builder from the function in which to
  /// instantiate the interface(s), the memory region that the interface(s) must
  /// reference, and a mapping between basic block IDs within the function and
  /// their respective control value, the latter of which which will be used to
  /// trigger the start of memory access groups in the interface(s).
  MemoryInterfaceBuilder(circt::handshake::FuncOp funcOp, Value memref,
                         const DenseMap<unsigned, Value> &ctrlVals)
      : funcOp(funcOp), memref(memref), ctrlVals(ctrlVals){};

  /// Adds an access port to an MC. The operation must be a load or store
  /// access to an MC. The operation must be tagged with the basic block it
  /// belongs to, which will be used to determine with which other MC ports this
  /// one belongs.
  void addMCPort(Operation *memOp);

  /// Adds an access port to a specific LSQ group. The operation must be a load
  /// or store access to an LSQ. The operation must be tagged with the basic
  /// block it belongs to.
  void addLSQPort(unsigned group, Operation *memOp);

  /// Instantiates appropriate memory interfaces for all the ports that were
  /// added to the builder so far. This may insert no interface, a single MC, a
  /// single LSQ, or both an MC and an LSQ depending on the set of recorded
  /// memory ports. On success, sets the data operand of recorded load access
  /// ports and returns instantiated interfaces through method arguments (which
  /// are set to nullptr if no interface of the type was created). Fails if the
  /// method could not determine memory inputs for the interface(s).
  LogicalResult
  instantiateInterfaces(OpBuilder &builder,
                        circt::handshake::MemoryControllerOp &mcOp,
                        circt::handshake::LSQOp &lsqOp);

  /// Returns results of load/store-like operations which are to be given as
  /// operands to a memory interface.
  static SmallVector<Value, 2> getMemResultsToInterface(Operation *memOp);

  /// Returns the result of a constant that serves as an MC control signal
  /// (indicating a non-zero number of stores in the block). Instantiates the
  /// constant operation in the IR after the provided none-typed control signal.
  static Value getMCControl(Value ctrl, unsigned numStores, OpBuilder &builder);

  /// Sets the data operand of a load-like operation, reusing the existing
  /// address operand.
  static void setLoadDataOperand(circt::handshake::LoadOpInterface loadOp,
                                 Value dataIn);

private:
  /// Wraps all inputs for instantiating an MC and/or an LSQ for the recorded
  /// memory ports. An empty list of inputs for the MC indicates that no MC is
  /// necessary for the recorded ports. The same is true for the LSQ.
  struct InterfaceInputs {
    /// Inputs for the MC.
    SmallVector<Value> mcInputs;
    /// List of basic block IDs for the MC.
    SmallVector<unsigned> mcBlocks;
    /// Inputs for the LSQ.
    SmallVector<Value> lsqInputs;
    /// List of group sizes for the MC.
    SmallVector<unsigned> lsqGroupSizes;
  };

  /// Groups a list of memory access ports by their group, which is a basic
  /// block ID for the MC and an abstract group number for the LSQ.
  using InterfacePorts = llvm::MapVector<unsigned, SmallVector<Operation *>>;

  /// Handshake function in which to instantiate memory interfaces.
  circt::handshake::FuncOp funcOp;
  /// Memory region that interface will reference.
  Value memref;
  /// Mapping between basic block ID and their respective entry control signal,
  /// for connecting the interface(s)'s control ports.
  DenseMap<unsigned, Value> ctrlVals;

  /// Memory access ports for the MC.
  InterfacePorts mcPorts;
  /// Number of loads to the MC.
  unsigned mcNumLoads = 0;
  /// Memory access ports for the LSQ.
  InterfacePorts lsqPorts;
  /// Number of loads to the LSQ.
  unsigned lsqNumLoads = 0;

  /// Determines the list of inputs for the memory interface(s) to instantiate
  /// from the sets of recorded ports. This performs no verification of the
  /// validity of the ports or their ordering. Fails if inputs could not be
  /// determined, in which case it is not possible to instantiate the
  /// interfaces.
  LogicalResult determineInterfaceInputs(InterfaceInputs &inputs,
                                         OpBuilder &builder);

  /// Returns the control signal for a specific block, as contained in the
  /// `ctrlVals` map. Produces an error on stderr and returns nullptr if no
  /// value exists for the block.
  Value getCtrl(unsigned block);

  /// For a provided memory interface and its memory ports, set the data operand
  /// of load-like operations with successive results of the memory interface.
  void addMemDataResultToLoads(InterfacePorts &ports, Operation *memIfaceOp);
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

#endif // DYNAMATIC_SUPPORT_HANDSHAKE_H
