//===- Attribute.h - Support for Dynamatic (operand) attributes -*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers to work with attributes in general but particulalry on attributes
// that are semantically set on IR operands rather than operations.
//
// It is impossible to set attributes on MLIR SSA values directly, therefore
// we opt to set "operand attributes" on the operation whose operands are being
// annotated instead. This implies that an operation may have the same type of
// operand attribute mutliple times if it has multiple operands. It may also
// contain multiple types of operand attributes for a single operand.
// Internally, operations keep track of which operand attributes are set on
// which of their operands using their index. The API defined in this file
// largely hides those cumbersome index management steps and exposes functions
// to manage operand attributes from the operand they correspond to directly.
//
// The process of defining operand attributes is straightforward. For any
// attribute type that one wishes to semantically set on an operand (the
// `OperandAttr`), one needs to additionally define a container attribute type
// that just serves as the map between operand indices and their corresponding
// attribute (the `ContainerAttr`). The `ContainerAttr` is the attribute that
// is set on the operand owner's top-level attribute dictionnary. It can be
// defined largely automatically using the TableGen `OperandContainerAttr` class
// defined in `HandshakeAttributes.td`. For any specific `OperandAttr`, a single
// `ContainerAttr` attribute is set on each operation having at least one
// operand annotated with an `OperandAttr`. The `ContainerAttr` and
// `OperandAttr` have to made aware of each other in TablegGen through `using`
// declarations. In `HandshakeAttributes.td`, this would look like the
// following.
//
// ```tablegen
//  def OperandAttr : Handshake_Attr<"Operand", "operand"> {
//    // other members
//
//    let extraClassDeclaration = [{
//      /// Container attribute corresponding to this operand attribute
//      using ContainerAttr = ::circt::handshake::ContainerAttr;
//    }];
//  }
//
//  def ContainerAttr : OperandContainerAttr<"Container", "container"> {
//    let summary = "Container attribute for OperandAttr";
//    let description = [{
//      Maps operands of an operation to their `OperandAttr` attribute, if
//      it exists. Never really needs to be interacted with by user code.
//    }];
//    let extraClassDeclaration = [{
//      /// Operand attribute corresponding to this container attribute
//      using OperandAttr = ::circt::handshake::OperandAttr;
//    }];
//  }
// ```
//
// Since the underlying attributes that stores operand attributes internally
// uses operand indices, modifying the list of operands in the presence of
// operand attributes will yield undefined behavior.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_ATTRIBUTE_H
#define DYNAMATIC_SUPPORT_ATTRIBUTE_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

/// Converts the attribute's name to an unsigned number. Asserts if the
/// attribute name doesn't represent a valid index (the verification function of
/// operand container attributes should prevent this from happening, but a user
/// may provide an incorrect attribute as template parameter to the get/set
/// functions).
static inline size_t toIdx(const mlir::NamedAttribute &attr) {
  std::string str = attr.getName().str();
  bool validNumber = std::all_of(str.begin(), str.end(),
                                 [](char c) { return std::isdigit(c); });
  assert(validNumber && "invalid index");
  return stoi(str);
}

namespace dynamatic {

/// Gets the attribute of the given type stored under the attribute mnemonic's
/// name on the operation, if it exists.
template <typename Attr>
static inline Attr getUniqueAttr(Operation *op) {
  return op->getAttrOfType<Attr>(Attr::getMnemonic());
}

/// Sets a unique attribute of the given type under the attribute mnemonic's
/// name on the operation.
template <typename Attr>
static inline void setUniqueAttr(Operation *op, Attr attr) {
  return op->setAttr(Attr::getMnemonic(), attr);
}

/// Copies an attribute of type `Attr` from the source operation to the
/// destination operation, if one exists with the same name as the attribute
/// type's mnemonic.
template <typename Attr>
static inline void copyAttr(Operation *srcOp, Operation *dstOp) {
  if (Attr attr = srcOp->getAttrOfType<Attr>(Attr::getMnemonic()))
    dstOp->setAttr(Attr::getMnemonic(), attr);
}

/// Copies attributes of all provided template types from the source operation
/// to the destination operation, if each exists with the same name as their
/// attribute type mnemonic.
template <typename FirstAttr, typename SecondAttr, typename... RestAttr>
static inline void copyAttr(Operation *srcOp, Operation *dstOp) {
  copyAttr<FirstAttr>(srcOp, dstOp);
  copyAttr<SecondAttr, RestAttr...>(srcOp, dstOp);
}

/// Casts the attribute's value to the template attribute type.
template <typename OperandAttr>
static inline OperandAttr toOperandAttr(const NamedAttribute &attr) {
  return attr.getValue().cast<OperandAttr>();
}

/// Returns the container attribute that corresponds the operand attribute of
/// the given template type, if it exists on the operation.
template <typename OperandAttr>
static inline typename OperandAttr::ContainerAttr
getContainerAttr(Operation *op) {
  return op->getAttrOfType<typename OperandAttr::ContainerAttr>(
      OperandAttr::ContainerAttr::getMnemonic());
}

/// Returns the operand's attribute of the given template type, if it exists.
template <typename OperandAttr>
OperandAttr getOperandAttr(OpOperand &oprd) {
  // The container attribute must be defined
  auto containerAttr = getContainerAttr<OperandAttr>(oprd.getOwner());
  if (!containerAttr)
    return nullptr;

  // Look for buffering properties attached to the channel
  unsigned oprdIdx = oprd.getOperandNumber();
  for (const NamedAttribute &attr : containerAttr.getOperandAttributes())
    if (toIdx(attr) == oprdIdx)
      return toOperandAttr<OperandAttr>(attr);
  return nullptr;
}

/// Returns all operand attributes of the given template type for the
/// operation's operands. The returned map is empty if no operand attributes of
/// this type exist.
template <typename OperandAttr>
mlir::DenseMap<OpOperand *, OperandAttr> getAllOperandAttrs(Operation *op) {
  DenseMap<Value, OperandAttr> props;
  // The container attribute must be defined
  auto containerAttr = getContainerAttr<OperandAttr>(op);
  if (!containerAttr)
    return props;

  // Map each result index to the corresponding value
  for (const NamedAttribute &attr : containerAttr.getOperandAttributes())
    props.insert(std::make_pair(op->getOperand(toIdx(attr)),
                                toOperandAttr<OperandAttr>(attr)));
  return props;
}

/// Sets the operand attribute of the given template type on the operand. If the
/// operand already had an attribute of this type, it is replaced.
template <typename OperandAttr>
void setOperandAttr(OpOperand &oprd, OperandAttr operandAttr) {
  Operation *owner = oprd.getOwner();
  size_t oprdIdx = oprd.getOperandNumber();
  MLIRContext *ctx = owner->getContext();

  // Create a list of attributes to store all the operand attributes of this
  // type
  SmallVector<NamedAttribute> operandAttrs;
  operandAttrs.push_back(NamedAttribute(
      StringAttr::get(ctx, std::to_string(oprdIdx)), operandAttr));

  if (auto attr = getContainerAttr<OperandAttr>(owner)) {
    // If the operation already had operand attributes of this type, we have to
    // recreate the full map
    for (const NamedAttribute &attr : attr.getOperandAttributes()) {
      if (toIdx(attr) != oprdIdx)
        operandAttrs.push_back(attr);
    }
  }
  // Set the container attribute
  owner->setAttr(OperandAttr::ContainerAttr::getMnemonic(),
                 OperandAttr::ContainerAttr::get(ctx, operandAttrs));
}

/// Removes the operand's attribute of the given template type, if it exists.
template <typename OperandAttr>
void removeOperandAttr(OpOperand &oprd) {
  Operation *owner = oprd.getOwner();
  size_t oprdIdx = oprd.getOperandNumber();
  MLIRContext *ctx = owner->getContext();

  // Create a list of attributes to store all the operand attributes of this
  // type
  SmallVector<NamedAttribute> operandAttrs;

  // Nothing to do if the container does not exist already
  auto attr = getContainerAttr<OperandAttr>(owner);
  if (!attr)
    return;

  // If the operation already had operand attributes of this type, we have to
  // recreate the full map
  bool foundAttribute = false;
  for (const NamedAttribute &attr : attr.getOperandAttributes()) {
    if (toIdx(attr) == oprdIdx)
      // Skip over the operand attribute being removed
      foundAttribute = true;
    else
      operandAttrs.push_back(attr);
  }
  if (!foundAttribute)
    // Nothing to do if the operand didn't have the attribute
    return;

  // Set the container attribute
  owner->setAttr(OperandAttr::ContainerAttr::getMnemonic(),
                 OperandAttr::ContainerAttr::get(ctx, operandAttrs));
}

/// Removes all operand attributes of the given template type for the
/// operation's operands.
template <typename OperandAttr>
void clearOperandAttr(Operation *op) {
  op->removeAttr(OperandAttr::ContainerAttr::getMnemonic());
}

} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_ATTRIBUTE_H