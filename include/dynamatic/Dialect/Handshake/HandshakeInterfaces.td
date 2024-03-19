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

def SOSTInterface : OpInterface<"SOSTInterface"> {
  let cppNamespace = "::dynamatic::handshake";
  let description = [{
      Sized Operation with Single Type (SOST).
      
      These are operations whose operands all have the same type and which have
      an integer size property, be it the number of operation operands (e.g.,
      for a merge) or the number of operation results (e.g., for a fork).
  }];

  let methods = [
    InterfaceMethod<[{
        Get the data type associated to the operation.
        The default implementation of this method simply returns the type of
        the first operation operand.
      }],
      "mlir::Type", "getDataType", (ins), "",
      [{
        auto concreteOp = mlir::cast<ConcreteOp>($_op);
        return concreteOp->getOperands().front().getType();
      }]
    >,
    InterfaceMethod<[{
        Get the size associated to the operation.
        The default implementation of this method simply returns the number of
        operation operands.
      }],
      "unsigned", "getSize", (ins), "",
      [{
        auto concreteOp = mlir::cast<ConcreteOp>($_op);
        return concreteOp->getNumOperands();
      }]
    >,
    InterfaceMethod<[{
        Determine whether the operation is a control operation.
        The default implementation of this method assumes that the operation
        is a control operation if and only if its associated data type is a
        NoneType.
      }],
      "bool", "sostIsControl", (ins), "",
      [{
        auto concreteOp = mlir::cast<ConcreteOp>($_op);
        // The operation is a control operation if its single data type is a 
        // NoneType.
        return concreteOp.getDataType().template isa<mlir::NoneType>();
      }]
    >,
    InterfaceMethod<[{
        Prints an SOST operation. If the `explicitSize` parameter is set to
        true, then the method prints the operation's size (in the SOST sense)
        between square brackets before printing the operation's operands,
        attributes, and data type.
      }],
      "void", "sostPrint", (ins 
        "mlir::OpAsmPrinter &": $printer, "bool": $explicitSize
      ), "", 
      [{
        auto concreteOp = mlir::cast<ConcreteOp>($_op);

        if (explicitSize) {
          printer << " [" << concreteOp.getSize() << "]";
        }
        printer << " " << concreteOp->getOperands();
        printer.printOptionalAttrDict(concreteOp->getAttrs());
        printer << " : " << concreteOp.getDataType();
      }]
    >
  ];

  let verify = [{
    auto concreteOp = mlir::cast<ConcreteOp>($_op);

    // SOST operation's size must be at least one
    if (concreteOp.getSize() < 1) {
      return concreteOp.emitOpError(
        "SOST operation's size must be at least 1, but has size ") 
        << concreteOp.getSize();
    }

    // SOST operation's operands must all have the same type
    auto dataType = concreteOp.getDataType();
    for (auto operand : concreteOp->getOperands())
      if (operand.getType() != dataType) 
        return concreteOp.emitOpError("SOST operation reports having data type ")
          << dataType << ", but one operand has type " << operand.getType();
    
    return mlir::success();
  }];
}

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
    InterfaceMethod<
        "Get the name of an operand.",
        "std::string", "getOperandName", (ins "unsigned" : $idx),
        "",
        [{
          // Return the default name of an operand.
          return "in" + std::to_string(idx);
        }]>,
    InterfaceMethod<
        "Get the name of a result.",
        "std::string", "getResultName", (ins "unsigned" : $idx),
        "",
        [{
          // Return the default name of a result.
          return "out" + std::to_string(idx);
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
        "bool",
        "isControl",
        (ins),
        "",
        [{
          // Implemented outside of interface due to dependency on
          // SOSTInterface being declared at the time where this method is
          // defined.
          return isControlOpImpl($_op);
        }]>];
}

def HasClock : NativeOpTrait<"HasClock">;

def SpeculationOpInterface : OpInterface<"SpeculationOpInterface"> {
  let cppNamespace = "::dynamatic::handshake";
  let description = [{
    Identifier for operations representing speculative units.
  }];
}

#endif //DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_INTERFACES

