//===- HandshakeOps.h - Handshake operations --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file originates from the CIRCT project (https://github.com/llvm/circt).
// It includes modifications made as part of Dynamatic.
//
//===----------------------------------------------------------------------===//
//
// This file defines convenience types for working with handshake operations.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_OPS_H
#define DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_OPS_H

#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/Any.h"

namespace dynamatic {
namespace handshake {

// Forward declaration needed by memory ports data structures.
class MCLoadOp;
class LSQLoadOp;
class MCStoreOp;
class LSQStoreOp;
class MemoryControllerOp;
class LSQOp;

} // end namespace handshake
} // end namespace dynamatic

namespace dynamatic {

/// Abstract base class for all memory ports. Each memory port denotes some kind
/// of (bi-)directional producer/consumer relationship between an operation
/// (referred to as the "port operation") and a memory interface (referred to as
/// the "memory interface"). The port operation may also be a memory interface.
///
/// The class hierarchy supports LLVM-style RTTI (i.e., isa/cast/dyn_cast) with
/// optional-value casting, see example below.
///
/// ```cpp
/// LoadPort loadPort (...);
/// SmallVector<MemoryPort> allPorts;
/// allPorts.push_back(loadPort);
/// std::optional<LoadPort> castedLoadPort = dyn_cast<LoadPort>(allPorts[0]);
/// assert(castedLoadPort);
/// std::optional<StorePort> notAStorePort = dyn_cast<StorePort>(allPorts[0]);
/// assert(!notAStorePort);
/// ```
class MemoryPort {
public:
  /// Kinds of memory ports (used for LLVM-style RTTI).
  enum class Kind {
    /// Control port.
    CONTROL,
    /// Load port (from load-like operation).
    LOAD,
    /// Load port (from dynamatic::handshake::MCLoadOp).
    MC_LOAD,
    /// Load port (from dynamatic::handshake::LSQLoadOp).
    LSQ_LOAD,
    /// Marker for last load-type port.
    LOAD_END,
    /// Store port (from store-like operation).
    STORE,
    /// Store port (from dynamatic::handshake::MCLStoreOp).
    MC_STORE,
    /// Store port (from dynamatic::handshake::LSQStoreOp).
    LSQ_STORE,
    /// Marker for last store-type port.
    STORE_END,
    /// MC load/store port (from dynamatic::handshake::MemoryControllerOp).
    MC_LOAD_STORE,
    /// LSQ load/store port (from dynamatic::handshake::LSQOp),
    LSQ_LOAD_STORE,
  };

  /// The operation producing the memory input(s) the port refers to.
  mlir::Operation *portOp;

  /// Deleted default constructor.
  MemoryPort() = delete;

  /// Default copy constructor.
  MemoryPort(const MemoryPort &other) = default;

  /// Returns the memory port's kind.
  Kind getKind() const { return kind; }

  explicit operator bool() const { return portOp != nullptr; }

  /// Virtual default destructor.
  virtual ~MemoryPort() = default;

protected:
  /// List of indices in the memory inputs/outputs of the memory interface the
  /// port refers to. Their meaning is port-kind-dependent.
  mlir::SmallVector<unsigned, 4> indices;

  /// Constructs a memory port "member-by-member".
  MemoryPort(mlir::Operation *portOp, mlir::ArrayRef<unsigned> indices,
             Kind kind);

private:
  /// Memory port's kind (used for LLVM-style RTTI).
  Kind kind;
};

/// Memory control port which may be associated with any operation type
/// (typically, a constant to indicate a number of stores in a block or a
/// control-only value from a control merge). It represents a single value in
/// the memory interface's inputs.
class ControlPort : public MemoryPort {
public:
  /// Constructs the control port from any operation whose single result ends up
  /// as the memory input indicated by the index.
  ControlPort(mlir::Operation *ctrlOp, unsigned ctrlInputIdx);

  /// Copy-constructor from abstract memory port for LLVM-style RTTI.
  ControlPort(const MemoryPort &memPort) : MemoryPort(memPort){};

  /// Returns the control operation the port is associated to.
  mlir::Operation *getCtrlOp() const { return portOp; }

  /// Returns the index of the control value in the memory interface's inputs.
  unsigned getCtrlInputIndex() const { return indices[0]; }

  /// Used by LLVM-style RTTI to establish `isa` relationships.
  static inline bool classof(const MemoryPort *port) {
    return port->getKind() == Kind::CONTROL;
  }
};

/// Memory load port associated with a load-like Handshake operation. It
/// represents two values in the memory interface's inputs/outputs.
/// 1. the address value produced by the port operation and consumed by the
/// memory interface, and
/// 2. the data value produced by the memory interface and consumed by the port
/// operation.
class LoadPort : public MemoryPort {
public:
  /// Default copy constructor.
  LoadPort(const LoadPort &other) = default;

  /// Copy-constructor from abstract memory port for LLVM-style RTTI.
  LoadPort(const MemoryPort &memPort) : MemoryPort(memPort){};

  /// Returns the load operation the port is associated to.
  dynamatic::handshake::LoadOpInterface getLoadOp() const;

  /// Returns the index of the load address value in the memory interface's
  /// inputs.
  unsigned getAddrInputIndex() const { return indices[0]; }

  /// Returns the index of the load data value in the memory interface's
  /// outputs.
  unsigned getDataOutputIndex() const { return indices[1]; }

  /// Used by LLVM-style RTTI to establish `isa` relationships.
  static inline bool classof(const MemoryPort *port) {
    Kind kind = port->getKind();
    return Kind::LOAD <= kind && kind < Kind::LOAD_END;
  }

protected:
  /// Constructs the load port from a load operation, the index of the load's
  /// address output in the memory interface's inputs, the index of the
  /// load's data input in the memory interface's outputs, and the specific load
  /// kind.
  LoadPort(dynamatic::handshake::LoadOpInterface loadOp, unsigned addrInputIdx,
           unsigned dataOutputIdx, Kind kind);
};

// Memory load port for memory controllers.
class MCLoadPort : public LoadPort {
public:
  /// Copy-constructor from abstract memory port for LLVM-style RTTI.
  MCLoadPort(const MemoryPort &memPort) : LoadPort(memPort){};

  /// Constructs the load port from an MC load operation, the index of the
  /// load's address output in the memory interface's inputs, and the index of
  /// the load's data input in the memory interface's outputs.
  MCLoadPort(dynamatic::handshake::MCLoadOp loadOp, unsigned addrInputIdx,
             unsigned dataOutputIdx);

  /// Returns the MC load operation the port is associated to.
  dynamatic::handshake::MCLoadOp getMCLoadOp() const;

  /// Used by LLVM-style RTTI to establish `isa` relationships.
  static inline bool classof(const MemoryPort *port) {
    return port->getKind() == Kind::MC_LOAD;
  }
};

// Memory load port for LSQs.
class LSQLoadPort : public LoadPort {
public:
  /// Copy-constructor from abstract memory port for LLVM-style RTTI.
  LSQLoadPort(const MemoryPort &memPort) : LoadPort(memPort){};

  /// Same semantics as the `LoadPort` constructor but works specifically with a
  /// load operation that connects to an LSQ.
  LSQLoadPort(dynamatic::handshake::LSQLoadOp loadOp, unsigned addrInputIdx,
              unsigned dataOutputIdx);

  /// Returns the LSQ load operation the port is associated to.
  dynamatic::handshake::LSQLoadOp getLSQLoadOp() const;

  /// Used by LLVM-style RTTI to establish `isa` relationships.
  static inline bool classof(const MemoryPort *port) {
    return port->getKind() == Kind::LSQ_LOAD;
  }
};

/// Memory store port associated with a store-like handshake operation. It
/// represents two values in the memory interface's inputs.
/// 1. the address value produced by the port operation and consumed by the
/// memory interface, and
/// 2. the data value produced by the port operation and consumed by the
/// memory interface.
class StorePort : public MemoryPort {
public:
  /// Default copy constructor.
  StorePort(const StorePort &other) = default;

  /// Copy-constructor from abstract memory port for LLVM-style RTTI.
  StorePort(const MemoryPort &memPort) : MemoryPort(memPort){};

  /// Returns the store operation the port is associated to.
  dynamatic::handshake::StoreOpInterface getStoreOp() const;

  /// Returns the index of the store address value in the memory interface's
  /// inputs.
  unsigned getAddrInputIndex() const { return indices[0]; }

  /// Returns the index of the store data value in the memory interface's
  /// inputs.
  unsigned getDataInputIndex() const { return indices[1]; }

  /// Used by LLVM-style RTTI to establish `isa` relationships.
  static inline bool classof(const MemoryPort *port) {
    Kind kind = port->getKind();
    return Kind::STORE <= kind && kind < Kind::STORE_END;
  }

protected:
  /// Constructs the store port from a store operation, the index of the
  /// store's address output in the memory interface's inputs (the store's data
  /// output is assumed to be at the next index), and the specific store kind.
  StorePort(dynamatic::handshake::StoreOpInterface storeOp,
            unsigned addrInputIdx, Kind kind);
};

// Memory store port for memory controllers.
class MCStorePort : public StorePort {
public:
  /// Copy-constructor from abstract memory port for LLVM-style RTTI.
  MCStorePort(const MemoryPort &memPort) : StorePort(memPort){};

  /// Same semantics as the `LoadPort` constructor but works specifically with a
  /// load operation that connects to an MC.
  MCStorePort(dynamatic::handshake::MCStoreOp mcStoreOp, unsigned addrInputIdx);

  /// Returns the MC store operation the port is associated to.
  dynamatic::handshake::MCStoreOp getMCStoreOp() const;

  /// Used by LLVM-style RTTI to establish `isa` relationships.
  static inline bool classof(const MemoryPort *port) {
    return port->getKind() == Kind::MC_STORE;
  }
};

// Memory store port for LSQs.
class LSQStorePort : public StorePort {
public:
  /// Copy-constructor from abstract memory port for LLVM-style RTTI.
  LSQStorePort(const MemoryPort &memPort) : StorePort(memPort){};

  /// Same semantics as the `LoadPort` constructor but works specifically with a
  /// load operation that connects to an LSQ.
  LSQStorePort(dynamatic::handshake::LSQStoreOp lsqStoreOp,
               unsigned addrInputIdx);

  /// Returns the LSQ store operation the port is associated to.
  dynamatic::handshake::LSQStoreOp getLSQStoreOp() const;

  /// Used by LLVM-style RTTI to establish `isa` relationships.
  static inline bool classof(const MemoryPort *port) {
    return port->getKind() == Kind::LSQ_STORE;
  }
};

/// Memory load/store port associated with a `dynamatic::handshake::LSQOp`,
/// which acts as a "middle-person" between individual load/store IR operations
/// and another memory interface (the one which this port is attached to). As
/// both a load port and a store port, it references 4 values through their
/// indices in the memory interface's inputs (3) and outputs (1).
/// 1. The load address value produced by the LSQ and consumed by the memory
/// interface (input).
/// 2. The load data value produced by the memory interface and consumed by the
/// LSQ (output).
/// 3. The store address value produced by the LSQ and consumed by the memory
/// interface (input).
/// 4. The store data value produced by the LSQ and consumed by the memory
/// interface (input).
class LSQLoadStorePort : public MemoryPort {
public:
  /// Constructs an LSQ load/store port from an LSQ operation, the index of the
  /// LSQ's load address output in the memory interface's inputs (the store
  /// address and store data inputs are assumed to follow), and the index of the
  /// LSQ's load data input in the memory interface's results.
  LSQLoadStorePort(dynamatic::handshake::LSQOp lsqOp, unsigned loadAddrInputIdx,
                   unsigned loadDataOutputIdx);

  /// Default copy constructor.
  LSQLoadStorePort(const LSQLoadStorePort &other) = default;

  /// Copy-constructor from abstract memory port for LLVM-style RTTI.
  LSQLoadStorePort(const MemoryPort &memPort) : MemoryPort(memPort){};

  /// Returns the LSQ the port is associated to.
  dynamatic::handshake::LSQOp getLSQOp() const;

  /// Returns the index of the load address value in the memory interface's
  /// inputs.
  unsigned getLoadAddrInputIndex() const { return indices[0]; }

  /// Returns the index of the load data value in the memory interface's
  /// outputs.
  unsigned getLoadDataOutputIndex() const { return indices[1]; }

  /// Returns the index of the store address value in the memory interface's
  /// inputs.
  unsigned getStoreAddrInputIndex() const { return indices[2]; }

  /// Returns the index of the store data value in the memory interface's
  /// inputs.
  unsigned getStoreDataInputIndex() const { return indices[3]; }

  /// Used by LLVM-style RTTI to establish `isa` relationships.
  static inline bool classof(const MemoryPort *port) {
    return port->getKind() == Kind::LSQ_LOAD_STORE;
  }
};

/// Memory load/store port associated with a
/// `dynamatic::handshake::MemoryControllerOp`, which acts as a "middle-person"
/// between an externally defined memory and another memory interface (the one
/// which this port is attached to). As both a load port and a store port, it
/// references 4 values through their indices in the memory interface's inputs
/// (1) and output (3).
/// 1. The load address value produced by the memory interface and consumed by
/// the LSQ (output).
/// 1. The load data value produced by the MC and consumed by the memory
/// interface (input).
/// 3. The store address value produced by the memory interface and consumed by
/// the LSQ (output).
/// 4. The store data value produced by the memory interface and consumed by the
/// LSQ (output).
class MCLoadStorePort : public MemoryPort {
public:
  /// Constructs an MC load/store port from an MC operation, the index of the
  /// MC's load address input in the memory interface outputs (the store address
  /// and store data inputs are assumed to follow), and the index of the memory
  /// MC's load data output in the memory interface's inputs.
  MCLoadStorePort(dynamatic::handshake::MemoryControllerOp mcOp,
                  unsigned loadAddrOutputIdx, unsigned loadDataInputIdx);

  /// Default copy constructor.
  MCLoadStorePort(const MCLoadStorePort &other) = default;

  /// Copy-constructor from abstract memory port for LLVM-style RTTI.
  MCLoadStorePort(const MemoryPort &memPort) : MemoryPort(memPort){};

  /// Returns the MC the port is associated to.
  dynamatic::handshake::MemoryControllerOp getMCOp() const;

  /// Returns the index of the load address value in the memory interface's
  /// outputs.
  unsigned getLoadAddrOutputIndex() const { return indices[0]; }

  /// Returns the index of the load data value in the memory interface's
  /// inputs.
  unsigned getLoadDataInputIndex() const { return indices[1]; }

  /// Returns the index of the store address value in the memory interface's
  /// outputs.
  unsigned getStoreAddrOutputIndex() const { return indices[2]; }

  /// Returns the index of the store data value in the memory interface's
  /// outputs.
  unsigned getStoreDataOutputIndex() const { return indices[3]; }

  /// Used by LLVM-style RTTI to establish `isa` relationships.
  static inline bool classof(const MemoryPort *port) {
    return port->getKind() == Kind::MC_LOAD_STORE;
  }
};

/// Represents a list of memory ports logically grouped together for a specific
/// memory interface. The semantics of a group are given by the memory interface
/// the group belongs to. A group may have a single control port and 0 or more
/// memory access ports (loads and stores) stored in the same order as the
/// memory interface's inputs. All ports in a group must map to a continuous
/// subrange of the memory interface's inputs.
class GroupMemoryPorts {
public:
  /// Optional control port for the group.
  std::optional<ControlPort> ctrlPort;
  /// List of load/store accesses to the memory interface, ordered the same as
  /// the latter's inputs.
  mlir::SmallVector<MemoryPort> accessPorts;

  /// Initializes a group's memory ports without a control port (and with no
  /// access ports).
  GroupMemoryPorts() = default;

  /// Initializes a group's memory ports with a control port (and with no access
  /// ports).
  GroupMemoryPorts(ControlPort ctrlPort);

  /// Whether the group's has a control port.
  inline bool hasControl() const { return ctrlPort.has_value(); }

  /// Computes the number of inputs in the asociated memory interface that map
  /// to this group's ports.
  unsigned getNumInputs() const;

  /// Computes the number of results in the asociated memory interface that map
  /// to this group's ports.
  unsigned getNumResults() const;

  /// Determines whether the group contains any port of the provided kinds.
  template <typename... PortKinds>
  bool hasAnyPort() const {
    if (ctrlPort && mlir::isa<PortKinds...>(*ctrlPort))
      return true;
    return llvm::any_of(accessPorts, [&](const MemoryPort &port) {
      return mlir::isa<PortKinds...>(port);
    });
  }

  /// Determines the number of ports of the provided kinds the group contains.
  template <typename... PortKinds>
  unsigned getNumPorts() const {
    unsigned count = 0;
    if (ctrlPort && mlir::isa<PortKinds...>(*ctrlPort))
      ++count;
    for (const MemoryPort &port : accessPorts) {
      if (mlir::isa<PortKinds...>(port))
        ++count;
    }
    return count;
  }
};

/// Represents all memory ports originating from a Handshake function for a
/// specific memory interface. Ports are aggregated in groups whose semantics
/// are given by the type of the memory interface. Groups are stored in the same
/// order as the memory interface's inputs. There may be 0 or more such groups.
/// Ports may also come from other memory interfaces.
class FuncMemoryPorts {
public:
  /// Memory interface associated with these ports.
  dynamatic::handshake::MemoryOpInterface memOp;
  /// List of groups which contain at least one input port to the memory
  /// interface, ordered the same as the latter's inputs.
  mlir::SmallVector<GroupMemoryPorts> groups;
  /// Ports to other memory interfaces (outside groups).
  mlir::SmallVector<MemoryPort> interfacePorts;
  /// Bitwidth of control signals.
  unsigned ctrlWidth = 0;
  /// Bitwidth of address signals.
  unsigned addrWidth = 0;
  /// Bitwidth of data signals.
  unsigned dataWidth = 0;

  /// Initializes a function's memory ports from the memory interface it
  /// corresponds to (and without any port).
  FuncMemoryPorts(dynamatic::handshake::MemoryOpInterface memOp)
      : memOp(memOp){};

  /// Returns the continuous subrange of the memory interface's inputs which a
  /// group (indicated by its index in the list) maps to.
  mlir::ValueRange getGroupInputs(unsigned groupIdx);

  /// Returns the continuous subrange of the memory interface's results which a
  /// group (indicated by its index in the list) maps to.
  mlir::ValueRange getGroupResults(unsigned groupIdx);

  /// Returns the number of groups attached to the memory interface.
  unsigned getNumGroups() { return groups.size(); }

  /// Determines whether the function contains any port of the provided kinds.
  template <typename... PortKinds>
  bool hasAnyPort() const {
    if (llvm::any_of(groups, [&](const GroupMemoryPorts &blockPorts) {
          return blockPorts.hasAnyPort<PortKinds...>();
        }))
      return true;
    return llvm::any_of(interfacePorts, [&](const MemoryPort &ifacePort) {
      return mlir::isa<PortKinds...>(ifacePort);
    });
  }

  /// Determines the number of ports of the provided kinds that the function
  /// contains.
  template <typename... PortKinds>
  unsigned getNumPorts() const {
    unsigned count = 0;
    for (const MemoryPort &ifacePort : interfacePorts) {
      if (mlir::isa<PortKinds...>(ifacePort))
        ++count;
    }
    for (const GroupMemoryPorts &blockPorts : groups)
      count += blockPorts.getNumPorts<PortKinds...>();
    return count;
  }
};

/// Smart-pointer around a `dynamatic::GroupMemoryPorts`, specializing it for
/// the `dynamatic::handshake::MemoryControllerOp` memory interface.
class MCBlock {
public:
  /// ID of the basic block the MC group corresponds to.
  unsigned blockID;

  /// Wraps a pointer to a `dynamatic::GroupMemoryPorts`.
  MCBlock(GroupMemoryPorts *group, unsigned blockID);

  /// Returns a reference to the underlying group.
  GroupMemoryPorts &operator*() { return *group; };

  /// Returns a pointer to the underlying group.
  GroupMemoryPorts *operator->() { return group; };

private:
  /// Underlying group of memory ports.
  GroupMemoryPorts *group;
};

/// Specialization of memory ports for a memory controller
/// (`dynamatic::handshake::MemoryControllerOp`), which may connect to an LSQ.
class MCPorts : public FuncMemoryPorts {
public:
  /// Initializes the ports for a memory controller (without any port).
  MCPorts(dynamatic::handshake::MemoryControllerOp mcOp);

  /// Returns the memory controller operation this refers to.
  dynamatic::handshake::MemoryControllerOp getMCOp() const;

  /// Returns the ports corresponding to a single LSQ groups.
  MCBlock getBlock(unsigned blockIdx);

  /// Returns a list of all LSQ groups, in definition order in the memory
  /// interface's inputs.
  mlir::SmallVector<MCBlock> getBlocks();

  /// Determines whether the memory controller connects to an LSQ.
  bool hasConnectionToLSQ() const { return !interfacePorts.empty(); }

  /// Returns the memory controller's LSQ ports (which must exist, check with
  /// `hasConnectionToLSQ`).
  LSQLoadStorePort getLSQPort() const {
    assert(hasConnectionToLSQ() && "no LSQ connected");
    return llvm::cast<LSQLoadStorePort>(interfacePorts.front());
  }
};

/// Smart-pointer around a `dynamatic::GroupMemoryPorts`, specializing it for
/// the `dynamatic::handshake::LSQOp` memory interface.
class LSQGroup {
public:
  /// Wraps a pointer to a `dynamatic::GroupMemoryPorts`.
  LSQGroup(GroupMemoryPorts *groups);

  /// Returns a reference to the underlying group.
  GroupMemoryPorts &operator*() { return *group; };

  /// Returns a pointer to the underlying group.
  GroupMemoryPorts *operator->() { return group; };

private:
  /// Underlying group of memory ports.
  GroupMemoryPorts *group;
};

/// Specialization of memory ports for an LSQ (`dynamatic::handshake::LSQOp`),
/// which may connect to a memory controller.
class LSQPorts : public FuncMemoryPorts {
public:
  /// Initializes the ports for an LSQ (without any port).
  LSQPorts(dynamatic::handshake::LSQOp lsqOp);

  /// Returns the memory controller operation this refers to.
  dynamatic::handshake::LSQOp getLSQOp() const;

  /// Returns the ports corresponding to a single LSQ groups.
  LSQGroup getGroup(unsigned groupIdx) { return LSQGroup(&groups[groupIdx]); }

  /// Returns a list of all LSQ groups, in definition order in the memory
  /// interface's inputs.
  mlir::SmallVector<LSQGroup> getGroups();

  /// Determines whether the LSQ connects to a memory controller.
  bool hasConnectionToMC() const { return !interfacePorts.empty(); }

  /// Returns the LSQ's memory controller ports (which must exist, check with
  /// `hasConnectionToMC`).
  MCLoadStorePort getMCPort() const {
    assert(hasConnectionToMC() && "no LSQ connected");
    return llvm::cast<MCLoadStorePort>(interfacePorts.front());
  }
};

/// Specifies how a handshake channel (i.e. a SSA value used once) may be
/// buffered. Backing data-structure for the ChannelBufPropsAttr attribute.
struct ChannelBufProps {
  /// Minimum number of transparent slots allowed on the channel (inclusive).
  unsigned minTrans;
  /// Maximum number of transparent slots allowed on the channel (inclusive).
  std::optional<unsigned> maxTrans;
  /// Minimum number of opaque slots allowed on the channel (inclusive).
  unsigned minOpaque;
  /// Maximum number of opaque slots allowed on the channel (inclusive).
  std::optional<unsigned> maxOpaque;
  /// Combinational delay (in ns) from the output port to the buffer's input, if
  /// a buffer is placed on the channel.
  double inDelay;
  /// Combinational delay (in ns) from the buffer's output to the input port, if
  /// a buffer is placed on the channel.
  double outDelay;
  /// Total combinational channel delay (in ns) if no buffer is placed on the
  /// channel.
  double delay;

  /// Simple constructor that takes the same parameters as the struct's members.
  /// By default, all the channel is "unconstrained" w.r.t. what kind of buffers
  /// can be placed and is assumed to have 0 delay.
  ChannelBufProps(unsigned minTrans = 0,
                  std::optional<unsigned> maxTrans = std::nullopt,
                  unsigned minOpaque = 0,
                  std::optional<unsigned> maxOpaque = std::nullopt,
                  double inDelay = 0.0, double outDelay = 0.0,
                  double delay = 0.0);

  /// Determines whether these buffering properties are satisfiable i.e.,
  /// whether it's possible to create a buffer that respects them.
  bool isSatisfiable() const;

  /// Determines whether these buffering properties forbid the placement of
  /// any buffer on the associated channel.
  bool isBufferizable() const;

  /// Computes member-wise equality.
  bool operator==(const ChannelBufProps &rhs) const;
};

/// Identifies the type of all ports in a memory interface's memory inputs by
/// backtracking their def-use chain till reaching specific operation types.
FuncMemoryPorts getMemoryPorts(dynamatic::handshake::MemoryOpInterface memOp);

} // namespace dynamatic

static inline std::string getMaxStr(std::optional<unsigned> optMax) {
  return optMax.has_value() ? (std::to_string(optMax.value()) + "]") : "inf]";
};

// Structs to enable LLVM-style RTTI for the memory port hierarchy.
namespace llvm {

/// Anything to a generic memory port.
template <typename T>
struct CastInfo<T, dynamatic::MemoryPort>
    : OptionalValueCast<T, dynamatic::MemoryPort> {};

/// Anything to a const generic memory port.
template <typename T>
struct CastInfo<T, const dynamatic::MemoryPort>
    : OptionalValueCast<T, const dynamatic::MemoryPort> {};

} // namespace llvm

/// Prints the buffering properties as two closed or semi-open intervals
/// (depending on whether maximums are defined), one for tranparent slots and
/// one for opaque slots.
template <typename Os>
Os &operator<<(Os &os, dynamatic::ChannelBufProps &props) {
  os << "{\n\ttransparent slots: [" << props.minTrans << ", "
     << getMaxStr(props.maxTrans) << "\n\topaque slots: [" << props.minOpaque
     << ", " << getMaxStr(props.maxOpaque) << "\n\tin/out delays: ("
     << props.inDelay << ", " << props.outDelay << ")"
     << "\n\ttotal delay: " << props.delay << "\n}\n";
  return os;
}

namespace mlir {
namespace OpTrait {
template <typename ConcreteType>
class HasClock : public TraitBase<ConcreteType, HasClock> {};

template <typename InterfaceType>
class HasParentInterface {
public:
  template <typename ConcreteType>
  class Impl : public TraitBase<ConcreteType,
                                HasParentInterface<InterfaceType>::Impl> {
  public:
    static LogicalResult verifyTrait(Operation *op) {
      if (llvm::isa_and_nonnull<InterfaceType>(op->getParentOp()))
        return success();

      // @mortbopet: What a horrible error message - however, there's no way to
      // report the interface name without going in and adjusting the tablegen
      // backend to also emit string literal names for interfaces.
      return op->emitOpError() << "expects parent op to be of the interface "
                                  "parent type required by the given op type";
    }
  };
};

} // namespace OpTrait

namespace affine {
struct DependenceComponent;
} // namespace affine
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h.inc"

#define GET_OP_CLASSES
#include "dynamatic/Dialect/Handshake/Handshake.h.inc"

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_OPS_H
