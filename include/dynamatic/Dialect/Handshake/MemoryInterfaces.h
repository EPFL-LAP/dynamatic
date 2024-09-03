//===- MemoryInterfaces.h - Memory interface helpers ------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares a couple data-structures and methods to work with
// Handshake memory interfaces (e.g., `handshake::MemoryControllerOp`,
// `handshake::LSQOp`).
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_DIALECT_HANDSHAKE_MEMORY_INTERFACES_H
#define DYNAMATIC_DIALECT_HANDSHAKE_MEMORY_INTERFACES_H

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Backedge.h"
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
  MemoryOpLowering(NameAnalysis &namer) : namer(namer) {};

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
/// rewiring" required to connect accesses to interfaces.
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
  /// reference, the memory start and control end signals, and a mapping between
  /// basic block IDs within the function and their respective control value,
  /// the latter of which which will be used to trigger the start of memory
  /// access groups in the interface(s).
  MemoryInterfaceBuilder(handshake::FuncOp funcOp, Value memref, Value memStart,
                         Value ctrlEnd,
                         const DenseMap<unsigned, Value> &ctrlVals)
      : funcOp(funcOp), memref(memref), memStart(memStart), ctrlEnd(ctrlEnd),
        ctrlVals(ctrlVals) {};

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
  LogicalResult instantiateInterfaces(OpBuilder &builder,
                                      handshake::MemoryControllerOp &mcOp,
                                      handshake::LSQOp &lsqOp);

  /// Instantiates appropriate memory interfaces for all the ports that were
  /// added to the builder so far using a pattern rewriter. See overload's
  /// documentation for more details.
  LogicalResult instantiateInterfaces(mlir::PatternRewriter &rewriter,
                                      handshake::MemoryControllerOp &mcOp,
                                      handshake::LSQOp &lsqOp);

  /// Returns results of load/store-like operations which are to be given as
  /// operands to a memory interface.
  static SmallVector<Value, 2> getMemResultsToInterface(Operation *memOp);

  /// Returns the result of a constant that serves as an MC control signal
  /// (indicating a non-zero number of stores in the block). Instantiates the
  /// constant operation in the IR after the provided control signal.
  static Value getMCControl(Value ctrl, unsigned numStores, OpBuilder &builder);

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
  handshake::FuncOp funcOp;
  /// Memory region that interface will reference.
  Value memref;
  /// Memory start signal.
  Value memStart;
  /// Control end signal, indicating that no more request will come.
  Value ctrlEnd;
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

  using FConnectLoad = std::function<void(handshake::LoadOpInterface, Value)>;

  /// For a provided memory interface and its memory ports, invoke the load
  /// connection callback for all load-like operations with successive results
  /// of the memory interface.
  void reconnectLoads(InterfacePorts &ports, Operation *memIfaceOp,
                      const FConnectLoad &connect);

  /// Internal implementation of the interface instantiation logic, taking an
  /// additional edge builder argument that was either created using a basic
  /// operation builder or a conversion pattern rewriter as well as a callback
  /// to connect the data input of loads to the newly created memory interfaces.
  LogicalResult instantiateInterfaces(OpBuilder &builder,
                                      BackedgeBuilder &edgeBuilder,
                                      const FConnectLoad &connect,
                                      handshake::MemoryControllerOp &mcOp,
                                      handshake::LSQOp &lsqOp);
};

/// Aggregates LSQ generation information to be passed to the DOT printer under
/// DOT attribute form or to the Chisel LSQ generator under JSON form.
struct LSQGenerationInfo {
  /// The LSQ for which generation information is being derived.
  handshake::LSQOp lsqOp;
  /// The name to give to the RTL module representing the LSQ.
  std::string name;
  /// Signals widths, for data and address buses.
  unsigned dataWidth, addrWidth;
  /// Number of groups, load ports, and store ports the LSQ connects to.
  unsigned numGroups, numLoads, numStores;
  /// Number of loads and store accesses per LSQ group.
  SmallVector<unsigned> loadsPerGroup, storesPerGroup;
  /// Index of first load and store port within each LSQ group (stored as a
  /// vector of vector to support the legacy Chisel LSQ generator, only the
  /// first value in each vector is meaningful).
  SmallVector<SmallVector<unsigned>> loadOffsets, storeOffsets;
  /// Overall indices for all load and store ports, split by LSQ group.
  SmallVector<SmallVector<unsigned>> loadPorts, storePorts;
  /// Depth of queues within the LSQ.
  unsigned depth = 16, depthLoad = 16, depthStore = 16, bufferDepth = 0;

  /// Derives generation information for the provided LSQ.
  LSQGenerationInfo(handshake::LSQOp lsqOp, StringRef name = "LSQ");

  /// Derives generation information for the provided LSQ, passed through its
  /// port information.
  LSQGenerationInfo(FuncMemoryPorts &ports, StringRef name = "LSQ");

private:
  /// Called by all constructor to derive generation information for an LSQ
  /// passed through its port information.
  void fromPorts(FuncMemoryPorts &ports);
};
}; // namespace dynamatic

#endif // DYNAMATIC_DIALECT_HANDSHAKE_MEMORY_INTERFACES_H