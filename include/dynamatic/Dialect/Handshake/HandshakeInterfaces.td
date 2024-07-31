//===- HandshakeInterfaces.td - Handshake interfaces -------*- tablegen -*-===//
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
// This is the definition file for the structured interface for Handshake ops.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_INTERFACES
#define DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_INTERFACES

include "mlir/IR/OpBase.td"

def MergeLikeOpInterface : OpInterface<"MergeLikeOpInterface"> {
  let cppNamespace = "::dynamatic::handshake";
  let description = [{
     Some handshake operations can have predecessors in other
     blocks. This is primarily useful for verification purposes during
     lowering from other dialect, such as the standard CDFG dialect.
  }];

  let methods = [
    InterfaceMethod<[{
        Returns an operand range over the data signals being merged.
      }],
      "mlir::OperandRange", "getDataOperands", (ins)
    >,
    InterfaceMethod<[{
        Returns the operation's result representing the selected data operand.
      }],
      "mlir::OpResult", "getDataResult", (ins)
    >,
  ];

  let verify = [{
    auto concreteOp = mlir::cast<ConcreteOp>($_op);

    auto operands = concreteOp.getDataOperands();

    if (!operands.size())
      return concreteOp.emitOpError("must have at least one data operand");

    mlir::Type resultType = $_op->getResult(0).getType();

    for (auto operand : operands)
      if (operand.getType() != resultType)
        return concreteOp.emitOpError("operand has type ") << operand.getType()
            << ", but result has type " << resultType;

    return mlir::success();
  }];
}

def MemoryOpInterface : OpInterface<"MemoryOpInterface"> {
  let cppNamespace = "::dynamatic::handshake";
  let description =
      [{"Allocate the memory to the meory map in the simulation. "}];

  let methods = [
    InterfaceMethod<
        "Returns the memref-typed SSA value that the interface references.",
        "::mlir::TypedValue<::mlir::MemRefType>", "getMemRef", (ins)>,
    InterfaceMethod<[{
        Returns the type of the memref-typed SSA value that the interface
        references.
        }],
        "::mlir::MemRefType", "getMemRefType", (ins),
        "",
        [{
          ConcreteOp concreteOp = cast<ConcreteOp>($_op);
          return concreteOp.getMemRef().getType().template cast<mlir::MemRefType>();
        }]>,
  ];
}

def BufferOpInterface : OpInterface<"BufferOpInterface"> {
  let cppNamespace = "::dynamatic::handshake";
  let description = [{
    Handshake operations which act as buffers with a strictly positive number
    of slots.
  }];

  let methods = [
    InterfaceMethod<
      "Returns the number of buffer slots.",
      "unsigned", "getSlots", (ins)
    >
  ];
}

def LoadOpInterface : OpInterface<"LoadOpInterface"> {
  let cppNamespace = "::dynamatic::handshake";
  let description = [{
    Handshake operations which act as load ports to a memory interface.
  }];

  let methods = [
    InterfaceMethod<
    "Returns the load port's address input (from the circuit).",
    "mlir::Value", "getAddressInput", (ins),
    "",
    [{
      ConcreteOp concreteOp = mlir::cast<ConcreteOp>($_op);
      return concreteOp.getAddress();
    }]>,
    InterfaceMethod<
    "Returns the load port's data input (from the memory interface).",
    "mlir::Value", "getDataInput", (ins),
    "",
    [{
      ConcreteOp concreteOp = mlir::cast<ConcreteOp>($_op);
      return concreteOp.getData();
    }]>,
    InterfaceMethod<
    "Returns the load port's address output (to the memory interface).",
    "mlir::Value", "getAddressOutput", (ins),
    "",
    [{
      ConcreteOp concreteOp = mlir::cast<ConcreteOp>($_op);
      return concreteOp.getAddressResult();
    }]>,
    InterfaceMethod<
    "Returns the load port's data output (to the circuit).",
    "mlir::Value", "getDataOutput", (ins),
    "",
    [{
      ConcreteOp concreteOp = mlir::cast<ConcreteOp>($_op);
      return concreteOp.getDataResult();
    }]>,
  ];
}

def StoreOpInterface : OpInterface<"StoreOpInterface"> {
  let cppNamespace = "::dynamatic::handshake";
  let description = [{
    Handshake operations which act as store ports to a memory interface.
  }];

  let methods = [
    InterfaceMethod<
    "Returns the store port's address input (from the circuit).",
    "mlir::Value", "getAddressInput", (ins),
    "",
    [{
      ConcreteOp concreteOp = mlir::cast<ConcreteOp>($_op);
      return concreteOp.getAddress();
    }]>,
    InterfaceMethod<
    "Returns the store port's data input (from the circuit).",
    "mlir::Value", "getDataInput", (ins),
    "",
    [{
      ConcreteOp concreteOp = mlir::cast<ConcreteOp>($_op);
      return concreteOp.getData();
    }]>,
    InterfaceMethod<
    "Returns the store port's address output (to the memory interface).",
    "mlir::Value", "getAddressOutput", (ins),
    "",
    [{
      ConcreteOp concreteOp = mlir::cast<ConcreteOp>($_op);
      return concreteOp.getAddressResult();
    }]>,
    InterfaceMethod<
    "Returns the store port's data output (to the memory interface).",
    "mlir::Value", "getDataOutput", (ins),
    "",
    [{
      ConcreteOp concreteOp = mlir::cast<ConcreteOp>($_op);
      return concreteOp.getDataResult();
    }]>,
  ];
}

def NamedIOInterface : OpInterface<"NamedIOInterface"> {
  let cppNamespace = "::dynamatic::handshake";
  let description =
      [{"Provides detailed names for the operands and results of an operation."}];

  let methods = [
    StaticInterfaceMethod<
        "Returns the default name of a specific operand.",
        "std::string", "getDefaultOperandName", (ins "unsigned" : $idx),
        "",
        [{
          return "ins_" + std::to_string(idx);
        }]>,
    StaticInterfaceMethod<
        "Returns the default name of a specific result.",
        "std::string", "getDefaultResultName", (ins "unsigned" : $idx),
        "",
        [{
          return "outs_" + std::to_string(idx);
        }]>,
    InterfaceMethod<
        "Returns the name of a specific operand.",
        "std::string", "getOperandName", (ins "unsigned" : $idx),
        "",
        [{
          ConcreteOp concreteOp = mlir::cast<ConcreteOp>($_op);

          // Operations which always have a single operand get a specific port
          // name for it
          if (concreteOp.template hasTrait<mlir::OpTrait::OneOperand>()) {
            assert(idx == 0 && "index too high");
            return "ins";
          }

          // Generic input name
          assert(idx < concreteOp->getNumOperands() && "index too high");
          return getDefaultOperandName(idx);
        }]>,
    InterfaceMethod<
        "Returns the name of a specific result.",
        "std::string", "getResultName", (ins "unsigned" : $idx),
        "",
        [{
          ConcreteOp concreteOp = mlir::cast<ConcreteOp>($_op);
          
          // Operations which always have a single result get a specific port
          // name for it
          if (concreteOp.template hasTrait<mlir::OpTrait::OneResult>()) {
            assert(idx == 0 && "index too high");
            return "outs";
          }

          // Generic output name
          assert(idx < concreteOp->getNumResults() && "index too high");
          return getDefaultResultName(idx);
        }]>
  ];
}

def ControlInterface : OpInterface<"ControlInterface"> {
  let cppNamespace = "::dynamatic::handshake";
  let description =
      [{"Provides information on whether this operation is a control operation."}];

  let methods = [
    InterfaceMethod<
      "Returns true if this operation is a control operation.",
      "bool", "isControl", (ins), "",
      "return false;"
    >];
}

def HasClock : NativeOpTrait<"HasClock">;

def SpeculationOpInterface : OpInterface<"SpeculationOpInterface"> {
  let cppNamespace = "::dynamatic::handshake";
  let description = [{
    Identifier for operations representing speculative units.
  }];
}

def SameExtraSignalsInterface : OpInterface<"SameExtraSignalsInterface"> {
  let cppNamespace = "::dynamatic::handshake";
  let description = [{
    Ensures that the extra signals of specific operand/result channels
    (`handshake::ChannelType`) are identical (even if the channels' respective
    data type may be different). 
  }];

  let methods = [
    InterfaceMethod<[{
        Returns the list of operand/result channels that should have the same
        extra signals.
      }], "::mlir::SmallVector<::mlir::TypedValue<::dynamatic::handshake::ChannelType>>",
        "getChannelsWithSameExtraSignals", (ins), "", [{
          return ::dynamatic::handshake::detail::getChannelsWithSameExtraSignals(
            $_op
          );
      }]>
  ];

  let verify = [{
    return ::dynamatic::handshake::detail::verifySameExtraSignalsInterface(
      $_op, ::mlir::cast<ConcreteOp>($_op).getChannelsWithSameExtraSignals()
    );
  }];
}

#endif //DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_INTERFACES
