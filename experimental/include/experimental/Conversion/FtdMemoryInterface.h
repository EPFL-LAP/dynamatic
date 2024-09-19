//===- FtdMemoryInterface.h - Memory interface helpers -*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Extended version of helper classes for the Memory Interce, in order to
// integrate what's needed for the fast-token-delivery pass
//
//===----------------------------------------------------------------------===//
//
//
#ifndef DYNAMATIC_CONVERSION_FTD_MEMORY_INTERFACE_H
#define DYNAMATIC_CONVERSION_FTD_MEMORY_INTERFACE_H

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/LLVM.h"
#include <set>

namespace dynamatic {
namespace experimental {
namespace ftd {

/// A `ProdConsMemDep` represents a memory dependancy between two blocks. In
/// particular, given the presence of an LSQ to prevent an hazard between two
/// memory accesses, the following three situations might happen:
///
/// - Write after Write (producer must complete before consumer so that the
/// final result is correct);
/// - Write after Read (producer must complete before consumer, so that the read
/// value is correct);
/// - Read after Write (producer bust complete beofre consumer, so that the read
/// value is correct).
///
/// The structure also highlights whether the relathionship is backward or not
/// (in case the two are in a loop)
struct ProdConsMemDep {
  Block *prodBb;
  Block *consBb;
  bool isBackward;

  ProdConsMemDep(Block *prod, Block *cons, bool backward)
      : prodBb(prod), consBb(cons), isBackward(backward) {}

  /// Print the dependency stored in the current relationship
  void printDependency();
};

/// A group represents all the memory operations belonging to the same basic
/// block which require the same LSQ. It contains a reference to the BB, a set
/// of predecessor in the dependence graph and a set of successors.
struct Group {
  // The BB the group defines
  Block *bb;

  // List of predecessors of the group
  std::set<Group *> preds;
  // List of successors of the group
  std::set<Group *> succs;

  // Constructor for the group
  Group(Block *b) : bb(b) {}

  // Relationship operator between groups
  bool operator<(const Group &other) const { return bb < other.bb; }

  /// Print the dependenices of the curent group
  void printDependenices();
};

class FtdMemoryInterfaceBuilder : public MemoryInterfaceBuilder {
public:
  using MemoryInterfaceBuilder::MemoryInterfaceBuilder;

  /// Instantiates appropriate memory interfaces for all the ports that were
  /// added to the builder so far. This may insert no interface, a single MC, a
  /// single LSQ, or both an MC and an LSQ depending on the set of recorded
  /// memory ports. On success, sets the data operand of recorded load access
  /// ports and returns instantiated interfaces through method arguments (which
  /// are set to nullptr if no interface of the type was created). Fails if the
  /// method could not determine memory inputs for the interface(s).
  /// This also adds a fork graph analogous to the group graph, and connects the
  /// fork nodes to the lsq inputs.
  LogicalResult instantiateInterfacesWithForks(
      OpBuilder &builder, handshake::MemoryControllerOp &mcOp,
      handshake::LSQOp &lsqOp, DenseSet<Group *> &groups,
      DenseMap<Block *, Operation *> &forksGraph, Value start,
      DenseSet<Operation *> &alloctionNetwork);

  /// Determines the list of inputs for the memory interface(s) to instantiate
  /// from the sets of recorded ports. This performs no verification of the
  /// validity of the ports or their ordering. Fails if inputs could not be
  /// determined, in which case it is not possible to instantiate the
  /// interfaces.
  /// This also adds a fork graph analogous to the group graph, and connects the
  /// fork nodes to the lsq inputs.
  LogicalResult determineInterfaceInputsWithForks(
      InterfaceInputs &inputs, OpBuilder &builder, DenseSet<Group *> &groups,
      DenseMap<Block *, Operation *> &forksGraphs, Value start,
      DenseSet<Operation *> &alloctionNetwork);
};

} // namespace ftd
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_CONVERSION_FTD_MEMORY_INTERFACE_H
