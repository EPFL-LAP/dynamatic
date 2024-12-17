//===- HandshakeTypeInterfaces.td - Handshake TypeInterfaces definition ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the type interfaces of the Handshake dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_TYPE_INTERFACES_TD
#define DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_TYPE_INTERFACES_TD

include "mlir/IR/OpBase.td"

def ExtraSignalsTypeInterface : TypeInterface<"ExtraSignalsTypeInterface"> {
  let cppNamespace = "dynamatic::handshake";
  let description = [{
    An interface for types with ExtraSignals to define and provide a
    default implementation of methods related to extra signals
    (getNumExtraSignals, getNumDownstreamExtraSignals, hasExtraSignal, etc)
  }];

  let methods = [
    // This method is auto-implemented if the type includes an `extraSignals`
    // argument of type `ExtraSignals` (defined in HandshakeTypes.td)
    InterfaceMethod<"Returns extra signals",
      "llvm::ArrayRef<ExtraSignal>", "getExtraSignals", (ins)>,
    InterfaceMethod<"Returns the number of extra signals",
      "unsigned", "getNumExtraSignals", (ins), /*methodBody=*/"",
      /*defaultImplementation=*/[{
        ConcreteType concreteType = mlir::cast<ConcreteType>($_type);
        return concreteType.getExtraSignals().size();
      }]>,
    InterfaceMethod<"Returns the number of downstream extra signals",
      "unsigned", "getNumDownstreamExtraSignals", (ins), /*methodBody=*/"",
      /*defaultImplementation=*/[{
        ConcreteType concreteType = mlir::cast<ConcreteType>($_type);
        return llvm::count_if(
            concreteType.getExtraSignals(),
            [](const ExtraSignal &extra) { return extra.downstream; });
      }]>,
    InterfaceMethod<"Returns the number of upstream extra signals",
      "unsigned", "getNumUpstreamExtraSignals", (ins), /*methodBody=*/"",
      /*defaultImplementation=*/[{
        ConcreteType concreteType = mlir::cast<ConcreteType>($_type);
        return concreteType.getNumExtraSignals() -
               concreteType.getNumDownstreamExtraSignals();
      }]>,
    InterfaceMethod<[{
        Returns an extra signal specified by the name
        Returns a failure if it is not found
      }],
      "std::optional<ExtraSignal>", "getExtraSignal",
      (ins "llvm::StringRef":$name), /*methodBody=*/"",
      /*defaultImplementation=*/[{
        ConcreteType concreteType = mlir::cast<ConcreteType>($_type);
        for (const ExtraSignal &extra : concreteType.getExtraSignals()) {
          if (extra.name == name)
            return extra;
        }
        return {};
      }]>,
    InterfaceMethod<"Returns whether the type has an extra signal",
      "bool", "hasExtraSignal", (ins "llvm::StringRef":$name),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        ConcreteType concreteType = mlir::cast<ConcreteType>($_type);
        return concreteType.getExtraSignal(name).has_value();
      }]>,
    // The implementation of this method largely depends on the type
    // Therefore, we don't provide the defaultImplementation on this
    InterfaceMethod<"Returns a new type with the extraSignal added",
      "mlir::Type", "addExtraSignal", (ins "const ExtraSignal &":$extraSignal)>,
  ];
}

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_TYPE_INTERFACES_TD
