//===- CFDFC.h - Control-Free DataFlow Circuit ------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares data structures and functions to extract and create CFDFCs
// (Choice-Free DataFlow Circuits) from a description of a Handshake function's
// archs and basic blocks.
//
// In this context, an arch is understood as an "edge" between two basic blocks.
// There exists an arch in a function between basic block X and Y if and only if
// there exists an operation in X that returns a result used by an operation in
// Y.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_CFDFC_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_CFDFC_H

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/ConstraintProgramming/ConstraintProgramming.h"
#include "dynamatic/Support/LLVM.h"
#include "experimental/Support/StdProfiler.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include <memory>
#include <vector>

namespace dynamatic {
namespace buffer {

/// A set of ArchBB's pointers with a deterministic iteration order.
using ArchSet = mlir::SetVector<dynamatic::experimental::ArchBB *>;
/// A set of basic block IDs with a deterministic iteration order.
using BBSet = mlir::SetVector<unsigned>;

struct CFDFC;
class CFDFCChannel;

/// A unit of a CFDFC together with the CFDFC channels that enter and leave it.
/// Units are owned by their CFDFC and only ever built and modified by it.
class CFDFCUnit {
public:
  /// Units should never be copied, as they're handled through pointers in
  /// CFDFC, so we remove its copy constructor and copy assignment.
  CFDFCUnit(const CFDFCUnit &) = delete;
  CFDFCUnit &operator=(const CFDFCUnit &) = delete;

  /// Returns the operation this unit stands for, or nullptr for the CFDFC's
  /// ghost entry unit.
  Operation *getOp() const { return op; }

  /// Returns the CFDFC channels entering the unit.
  ArrayRef<const CFDFCChannel *> getInChannels() const { return inChannels; }

  /// Returns the CFDFC channels leaving the unit.
  ArrayRef<const CFDFCChannel *> getOutChannels() const { return outChannels; }

  /// Returns the consumers of the unit's out channels.
  ArrayRef<const CFDFCUnit *> getSuccessors() const { return succs; }

  /// Returns the producers of the unit's in channels.
  ArrayRef<const CFDFCUnit *> getPredecessors() const { return preds; }

private:
  /// The CFDFC is the only builder of units and channels.
  friend struct CFDFC;

  explicit CFDFCUnit(Operation *op) : op(op) {}

  Operation *op;
  SmallVector<CFDFCChannel *, 4> inChannels;
  SmallVector<CFDFCChannel *, 4> outChannels;

  SmallVector<CFDFCUnit *, 4> succs;
  SmallVector<CFDFCUnit *, 4> preds;
};

/// A channel of a CFDFC, from the unit of its producer to the unit of its
/// consumer. Channels are owned by their CFDFC and only ever built and modified
/// by it.
class CFDFCChannel {
public:
  /// Channels should never be copied, as they're handled through pointers in
  /// CFDFC, so we remove its copy constructor and copy assignment.
  CFDFCChannel(const CFDFCChannel &) = delete;
  CFDFCChannel &operator=(const CFDFCChannel &) = delete;

  /// Returns the MLIR value this channel stands for.
  Value getValue() const { return value; }

  /// Returns the unit of the channel's producer.
  const CFDFCUnit *getSrc() const { return src; }

  /// Returns the unit of the channel's consumer.
  const CFDFCUnit *getDst() const { return dst; }

  /// Returns whether the channel is a "CFDFC backedge" (see
  /// CFDFC::isCFDFCBackedge).
  bool isBackedge() const { return backedge; }

private:
  /// The CFDFC is the only builder of units and channels.
  friend struct CFDFC;

  CFDFCChannel(Value value, CFDFCUnit *src, CFDFCUnit *dst, bool backedge)
      : value(value), src(src), dst(dst), backedge(backedge) {}

  Value value;
  CFDFCUnit *src;
  CFDFCUnit *dst;
  bool backedge;
};

/// The data of a CFDFC apart from its units and channels.
struct CFDFCInfo {
  /// The list of basic blocks that make up the CFDFC.
  mlir::SetVector<unsigned> cycle;
  /// Number of executions of the CFDFC.
  unsigned numExecs = 0;

  /// (Available after placement) The achieved throughput after the placement
  double throughput = 0.0;
  /// (Available after placement) The number of tokens held by each placed
  /// buffer.
  mlir::DenseMap<Operation *, double> bufferOccupancy;
  /// (Available after placement) The number of tokens per channel. After
  /// instantiating the buffers, this value is cleared and transferred to the
  /// buffer(s).
  mlir::DenseMap<Value, double> channelOccupancy;
};

/// Represents a CFDFC i.e., a set of control-free units and channels from a
/// dataflow circuit accompanied by the number of times it was executed.
/// The units and channels form a directed graph.
///
/// A CFDFC owns its units and channels, which keep their address for the
/// CFDFC's whole lifetime, including when the CFDFC is moved. CFDFCs cannot be
/// copied implicitly; use `clone` to duplicate one.
struct CFDFC : public CFDFCInfo {
private:
  using UnitMap = llvm::MapVector<Operation *, std::unique_ptr<CFDFCUnit>>;
  using ChannelMap = llvm::MapVector<Value, std::unique_ptr<CFDFCChannel>>;

  static const CFDFCUnit *getUnitPtr(const UnitMap::value_type &entry) {
    return entry.second.get();
  }
  static const CFDFCChannel *
  getChannelPtr(const ChannelMap::value_type &entry) {
    return entry.second.get();
  }

public:
  using unit_iterator =
      llvm::mapped_iterator<UnitMap::const_iterator, decltype(&getUnitPtr)>;
  using channel_iterator = llvm::mapped_iterator<ChannelMap::const_iterator,
                                                 decltype(&getChannelPtr)>;

  /// Constructs a CFDFC from a set of selected archs and basic blocks in the
  /// function using the traditional channel selection.
  static CFDFC create(handshake::FuncOp funcOp, ArchSet &archs,
                      unsigned numExec);

  /// Constructs a CFDFC from a set of selected archs and basic blocks in the
  /// function using the FTD-specific channel selection, which additionally
  /// requires backedges to be specified.
  static CFDFC createFTD(handshake::FuncOp funcOp, ArchSet &archs,
                         unsigned numExec,
                         const DenseSet<Value> &backwardChannels);

  CFDFC(const CFDFC &) = delete;
  CFDFC &operator=(const CFDFC &) = delete;
  CFDFC(CFDFC &&) = default;
  CFDFC &operator=(CFDFC &&) = default;

  /// Returns a copy of the CFDFC with its own units and channels.
  CFDFC clone() const;

  //===--------------------------------------------------------------------===//
  // Graph access
  //===--------------------------------------------------------------------===//

  /// Returns the units of the CFDFC, in insertion order.
  llvm::iterator_range<unit_iterator> getCFDFCUnits() const {
    return {unit_iterator(units.begin(), &getUnitPtr),
            unit_iterator(units.end(), &getUnitPtr)};
  }
  /// Returns the channels of the CFDFC, in insertion order.
  llvm::iterator_range<channel_iterator> getCFDFCChannels() const {
    return {channel_iterator(channels.begin(), &getChannelPtr),
            channel_iterator(channels.end(), &getChannelPtr)};
  }

  /// Returns the operations of the CFDFC's units, in insertion order.
  auto getUnits() const { return llvm::make_first_range(units); }
  /// Returns the values of the CFDFC's channels, in insertion order.
  auto getChannels() const { return llvm::make_first_range(channels); }
  /// Returns the values of the CFDFC's backedges, in insertion order. In FTD
  /// mode, see also getFTDBackedges.
  auto getBackedges() const {
    return llvm::map_range(
        llvm::make_filter_range(
            getCFDFCChannels(),
            [](const CFDFCChannel *channel) { return channel->isBackedge(); }),
        [](const CFDFCChannel *channel) { return channel->getValue(); });
  }

  size_t getNumUnits() const { return units.size(); }
  size_t getNumChannels() const { return channels.size(); }
  size_t getNumBackedges() const;

  /// (FTD only) Backward channels between units of the CFDFC that are not
  /// channels of the CFDFC because they do not follow an edge of the CFDFC's
  /// block cycle. They are not part of the graph.
  ArrayRef<Value> getFTDBackedges() const { return ftdBackedges; }

  /// Returns the unit standing for the operation, or nullptr if the operation
  /// is not in the CFDFC.
  const CFDFCUnit *lookup(Operation *op) const;
  /// Returns the channel standing for the value, or nullptr if the value is not
  /// in the CFDFC.
  const CFDFCChannel *lookup(Value value) const;

  bool contains(Operation *op) const { return units.count(op); }
  bool contains(Value value) const { return channels.count(value); }

  /// Returns whether the value is a backedge of the CFDFC.
  bool isBackedge(Value value) const;

  /// Returns the ghost entry unit, which is not part of the CFDFC's units, with
  /// an edge to every unit and no incoming edge. Algorithms that start from a
  /// single entry node (e.g., llvm::scc_iterator) reach the whole CFDFC from
  /// it.
  const CFDFCUnit *getEntryNode() const { return entry.get(); }

  /// Returns the strongly connected components of the CFDFC, each as the list
  /// of the operations of its units, in reverse topological order (as
  /// llvm::scc_iterator).
  std::vector<SmallVector<Operation *>> getSCCs() const;

  //===--------------------------------------------------------------------===//
  // Graph modification
  //===--------------------------------------------------------------------===//

  /// Removes a channel from the CFDFC. Its producer and consumer stay in the
  /// CFDFC.
  void removeChannel(Value value);

  /// Adds units placed on a channel of the CFDFC (e.g., buffers) to the CFDFC.
  /// The units must form a chain from the channel's producer to its original
  /// consumer: the first unit consumes `value` and each unit's first result
  /// is the only input of the next unit or of the original consumer. The
  /// channel then ends at the first unit, and the results of the units become
  /// new, non-backedge channels.
  void insertUnitsOnChannel(Value value, ArrayRef<Operation *> newOps);

  //===--------------------------------------------------------------------===//
  // Helpers
  //===--------------------------------------------------------------------===//

  /// Returns whether the channel between the two blocks has a corresponding
  /// edge in this CFDFC's CFG cycle.
  bool isCFGCompliant(unsigned srcBB, unsigned dstBB) const;

  // Determines whether the channel is a "CFDFC backedge" i.e., the first
  // channel along a sequence of backedges from a source block to a destination
  // block. The distinction is important for the buffer placement MILP, which
  // uses backedges to determine where to insert "tokens" in the circuit.
  static bool isCFDFCBackedge(Value val);

private:
  /// Builds an empty CFDFC, with only its ghost entry unit.
  CFDFC();

  /// Builds the block cycle from the selected archs and collects the units
  /// belonging to it. Channels are collected afterwards by one of the
  /// collect functions.
  CFDFC(handshake::FuncOp funcOp, ArchSet &archs, unsigned numExec);

  /// Adds a unit for the operation, which must not be in the CFDFC yet.
  CFDFCUnit *addUnit(Operation *op);

  /// Adds a channel for the value, which must not be in the CFDFC yet, between
  /// two units of the CFDFC.
  void addChannel(Value value, CFDFCUnit *src, CFDFCUnit *dst, bool isBackedge);

  /// Adds a channel for the value between the units of its producer and of its
  /// consumer, which must be in the CFDFC.
  void addChannel(Value value, bool isBackedge);

  /// Collects the channels and backedges of the units using the traditional
  /// channel selection.
  void collectChannels();

  /// Collects the channels and backedges of the units using the FTD-specific
  /// channel selection.
  void collectChannelsFTD(const DenseSet<Value> &backwardChannels);

  UnitMap units;
  ChannelMap channels;
  /// Not part of `CFDFCInfo` because it is set together with the channels;
  /// `clone` copies it explicitly.
  SmallVector<Value> ftdBackedges;
  /// "Ghost" entry node, heap-allocated like the other units.
  std::unique_ptr<CFDFCUnit> entry;
};

/// Represents a union of CFDFCs. Its blocks, units, channels, and backedges are
/// unions of the corresponding elements from all the CFDFCs that make it up. A
/// CFDFC union, unlike a regular CFDFC, does not contain a number of executions
/// since it does not make sense in the context of a union.
struct CFDFCUnion {
  /// The individual CFDFCs the union is made up of.
  SmallVector<CFDFC *> cfdfcs;
  /// The set basic blocks that make up the CFDFC union.
  mlir::SetVector<unsigned> blocks;
  /// Units (i.e., MLIR operations) in the CFDFC union.
  mlir::SetVector<Operation *> units;
  /// Channels (i.e., MLIR values) in the CFDFC union.
  mlir::SetVector<Value> channels;
  /// Backedges in the CFDFC union.
  mlir::SetVector<Value> backedges;

  /// Constructs the CFDFC union from an array of individual CFDFCs. The CFDFCs
  /// must outlive the union, which stores pointers to them.
  CFDFCUnion(ArrayRef<CFDFC *> cfdfcs);
};

/// Computes a set of CFDFC unions from a list of CFDFCs where each CFDFC in
/// each union shares at least one block with at least one other CFDFC of the
/// same union, and any two CFDFCs in different unions are completely disjoint.
/// Internally uses a disjoint-set data-structure for fast computation times and
/// low memory footprint.
void getDisjointBlockUnions(ArrayRef<CFDFC *> cfdfcs,
                            std::vector<CFDFCUnion> &unions);

/// Extracts the most frequently executed CFDFC from the Handshake function
/// described by the provided archs and basic blocks. The function internally
/// expresses the CFDFC extraction problem as an MILP that is solved bu Gurobi
/// (hence building the project with Gurobi is required to use this function).
/// On successfull extraction, succeeds and sets the last two arguments with,
/// respectively, the set of archs included in the extracted CFDFC and the
/// number of executions of the latter. When no CFDFC could be extracted,
/// succeeds but sets the number of executions to 0. On failure, and if
/// `milpStat` is not nullptr, the Gurobi status is saved in it.
LogicalResult extractCFDFC(handshake::FuncOp funcOp, ArchSet &archs, BBSet &bbs,
                           ArchSet &selectedArchs, unsigned &numExec,
                           CPSolver::SolverKind solverKind,
                           const std::string &logPath = "",
                           int *milpStat = nullptr);

} // namespace buffer
} // namespace dynamatic

namespace llvm {
template <>
struct GraphTraits<const dynamatic::buffer::CFDFCUnit *> {
  using NodeRef = const dynamatic::buffer::CFDFCUnit *;
  using ChildIteratorType = ArrayRef<NodeRef>::iterator;
  static NodeRef getEntryNode(NodeRef unit) { return unit; }
  static ChildIteratorType child_begin(NodeRef unit) {
    return unit->getSuccessors().begin();
  }
  static ChildIteratorType child_end(NodeRef unit) {
    return unit->getSuccessors().end();
  }
};

/// Walking a CFDFC starts from its ghost entry unit, from which every unit is
/// guaranteed to be reachable.
template <>
struct GraphTraits<const dynamatic::buffer::CFDFC *>
    : public GraphTraits<const dynamatic::buffer::CFDFCUnit *> {
  using nodes_iterator = dynamatic::buffer::CFDFC::unit_iterator;
  static NodeRef getEntryNode(const dynamatic::buffer::CFDFC *cfdfc) {
    return cfdfc->getEntryNode();
  }
  static nodes_iterator nodes_begin(const dynamatic::buffer::CFDFC *cfdfc) {
    return cfdfc->getCFDFCUnits().begin();
  }
  static nodes_iterator nodes_end(const dynamatic::buffer::CFDFC *cfdfc) {
    return cfdfc->getCFDFCUnits().end();
  }
  static unsigned size(const dynamatic::buffer::CFDFC *cfdfc) {
    return cfdfc->getNumUnits();
  }
};
} // namespace llvm

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_CFDFC_H
