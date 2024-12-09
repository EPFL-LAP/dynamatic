//===- HandshakeTypeInterfaces.td - Handshake TypeInterfaces definition ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines Handshake TypeInterfaces in Tablegen.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_TYPE_INTERFACES_TD
#define DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_TYPE_INTERFACES_TD

include "mlir/IR/OpBase.td"

def ExtraSignalsTypeInterface : TypeInterface<"ExtraSignalsTypeInterface"> {
  let cppNamespace = "dynamatic::handshake";
  let description = [{
    An interface for types with ExtraSignals
  }];

  // methodBody: the method body of the interface itself
  // defaultImplementation: the default body of the method of the type implementing this interface
  // ConcreteType: the type (template) implementing this interface
  let methods = [
    InterfaceMethod<"Returns the number of extra signals",
      "unsigned", "getNumExtraSignals", (ins), /*methodBody=*/"", /*defaultImplementation=*/[{
        ConcreteType concreteType = mlir::cast<ConcreteType>($_type);
        return concreteType.getExtraSignals().size();
      }]>,
    InterfaceMethod<"Returns the number of extra signals",
      "unsigned", "getNumDownstreamExtraSignals", (ins), /*methodBody=*/"", /*defaultImplementation=*/[{
        ConcreteType concreteType = mlir::cast<ConcreteType>($_type);
        return llvm::count_if(concreteType.getExtraSignals(), [](const ExtraSignal &extra) {
          return extra.downstream;
        });
      }]>,
    InterfaceMethod<"Returns the number of extra signals",
      "unsigned", "getNumUpstreamExtraSignals", (ins), /*methodBody=*/"", /*defaultImplementation=*/[{
        ConcreteType concreteType = mlir::cast<ConcreteType>($_type);
        return concreteType.getNumExtraSignals() - concreteType.getNumDownstreamExtraSignals();
      }]>,
    InterfaceMethod<[{
        Returns an extra signal specified by the name
        Returns a failure if it is not found
      }],
      "mlir::FailureOr<ExtraSignal>", "getExtraSignalByName",
      (ins "const std::string &":$name), /*methodBody=*/"", /*defaultImplementation=*/[{
        ConcreteType concreteType = mlir::cast<ConcreteType>($_type);
        for (const ExtraSignal &extra : concreteType.getExtraSignals()) {
          if (extra.name == name)
            return extra;
        }
        return mlir::failure();
      }]>,
    InterfaceMethod<"Returns whether the type has an extra signal specified by the name",
      "bool", "hasExtraSignal", (ins "const std::string &":$name), /*methodBody=*/"", /*defaultImplementation=*/[{
        ConcreteType concreteType = mlir::cast<ConcreteType>($_type);
        return !mlir::failed(concreteType.getExtraSignalByName(name));
      }]>,
    // The implementation of these methods largely depend on the type
    // Thus we don't provide the defaultImplementation on them
    InterfaceMethod<"Returns a new type with the extraSignal added",
      "mlir::Type", "addExtraSignal", (ins "const ExtraSignal &":$extraSignal)>,
    InterfaceMethod<"Returns a new type without the extraSignal",
      "mlir::Type", "removeExtraSignal", (ins "const std::string &":$name)>,
  ];
}

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_TYPE_INTERFACES_TD
