//===- Attribute.h - Support for Dynamatic (operand) attributes -*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers to work with attributes in general but particulalry on "dialect
// attributes" (see definition below) and attributes that are semantically set
// on IR operands rather than operations.
//
//===- Dialect attributes -------------------------------------------------===//
//
// Whenever we reference dialect attributes in this file, we mean a discardable
// attribute whose name is structured as follows.
//
// <dialect-name>.<attribute-mnemonic>
//
// In general, the dialect whose name prefixes the attribute name is the dialect
// which the attribute is registered to, though some functions in this file take
// a dialect class as a template parameter (which defaults to the Handshake
// dialect) that allows to customize this behavior.
//
// Functions working on dialect attributes are useful when one wants to attach a
// specific attribute type to an operation and when it only makes sense for at
// most one instance of the attribute type to be attached to the operation at
// any given time. In this scenario, client code does not have to care about the
// attribute's name when getting or setting the attribute since it is derived
// from the attribute type itself.
//
//===- Operand attributes -------------------------------------------------===//
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
//      using ContainerAttr = ::dynamatic::handshake::ContainerAttr;
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
//      using OperandAttr = ::dynamatic::handshake::OperandAttr;
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

#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Operation.h"

namespace dynamatic {

namespace detail {
/// Returns a discardable attribute name made up of the dialect's namespace and
/// the attribute mnemonic separated by a dot.
template <typename Dialect, typename Attr>
inline std::string getDialectAttrName() {
  return Dialect::getDialectNamespace().str() + "." + Attr::getMnemonic().str();
}

/// Returns a discardable attribute name made up of the dialect's namespace that
/// the attribute is registered to and the attribute mnemonic separated by a
/// dot.
template <typename Attr>
inline std::string getDialectAttrName(Attr attr) {
  return attr.getDialect().getNamespace().str() + "." +
         Attr::getMnemonic().str();
}

/// Converts the attribute's name to an unsigned number. Asserts if the
/// attribute name doesn't represent a valid index (the verification function of
/// operand container attributes should prevent this from happening, but a user
/// may provide an incorrect attribute as template parameter to the get/set
/// functions).
size_t toIdx(const mlir::NamedAttribute &attr);

/// Casts the named attribute's value to the template attribute type.
template <typename OperandAttr>
inline OperandAttr toOperandAttr(const NamedAttribute &attr) {
  return cast<OperandAttr>(attr.getValue());
}
} // namespace detail

/// Attempts to retrieve a dialect attribute of the given type. Returns
/// `nullptr` if no such attribute exists.
template <typename Attr, typename Dialect = handshake::HandshakeDialect>
inline Attr getDialectAttr(Operation *op) {
  std::string name = detail::getDialectAttrName<Dialect, Attr>();
  return mlir::dyn_cast_if_present<Attr>(op->getDiscardableAttr(name));
}

/// Sets a dialect attribute of the given type.
template <typename Attr>
inline void setDialectAttr(Operation *op, Attr attr) {
  std::string name = detail::getDialectAttrName<Attr>(attr);
  op->setDiscardableAttr(name, attr);
}

/// Sets a dialect attribute of the given type. Arguments are forwarded to the
/// attribute type's `get` function.
template <typename Attr, typename... Args>
inline void setDialectAttr(Operation *op, Args... args) {
  Attr attr = Attr::get(std::forward<Args>(args)...);
  std::string name = detail::getDialectAttrName<Attr>(attr);
  op->setDiscardableAttr(name, attr);
}

/// Removes a dialect attribute of the given type. If the attribute existed,
/// returns it; otherwise returns `nullptr`.
template <typename Attr, typename Dialect = handshake::HandshakeDialect>
inline Attribute removeDialectAttr(Operation *op) {
  std::string name = detail::getDialectAttrName<Dialect, Attr>();
  return op->removeDiscardableAttr(name);
}

/// Attempts to copy a dialect attribute of the given type from the source
/// operation to the destination operation. Returns whether such an attribute
/// exists on the source operation (and whether it was subsequently copied).
template <typename Attr, typename Dialect = handshake::HandshakeDialect>
inline bool copyDialectAttr(Operation *srcOp, Operation *dstOp) {
  std::string name = detail::getDialectAttrName<Dialect, Attr>();
  if (Attr attr = srcOp->getAttrOfType<Attr>(name)) {
    dstOp->setDiscardableAttr(name, attr);
    return true;
  }
  return false;
}

/// Attempts to retrieve tge container dialect attribute that corresponds to the
/// operand attribute of the given type. Returns `nullptr` of no such attribute
/// exists.
template <typename OperandAttr, typename Dialect = handshake::HandshakeDialect>
inline typename OperandAttr::ContainerAttr getContainerAttr(Operation *op) {
  return getDialectAttr<typename OperandAttr::ContainerAttr, Dialect>(op);
}

/// Sets the container dialect attribute that corresponds to the operand
/// attribute of the given type.
template <typename OperandAttr>
inline void
setContainerAttr(Operation *op,
                 typename OperandAttr::ContainerAttr containerAttr) {
  setDialectAttr<typename OperandAttr::ContainerAttr>(op, containerAttr);
}

/// Returns the operand's attribute of the given template type, if it exists.
template <typename OperandAttr, typename Dialect = handshake::HandshakeDialect>
OperandAttr getOperandAttr(OpOperand &oprd) {
  // The container attribute must be defined
  auto containerAttr = getContainerAttr<OperandAttr, Dialect>(oprd.getOwner());
  if (!containerAttr)
    return nullptr;

  // Look for buffering properties attached to the channel
  unsigned oprdIdx = oprd.getOperandNumber();
  for (const NamedAttribute &attr : containerAttr.getOperandAttributes())
    if (detail::toIdx(attr) == oprdIdx)
      return detail::toOperandAttr<OperandAttr>(attr);
  return nullptr;
}

/// Returns all operand attributes of the given template type for the
/// operation's operands. The returned map is empty if no operand attributes of
/// this type exist.
template <typename OperandAttr, typename Dialect = handshake::HandshakeDialect>
mlir::DenseMap<OpOperand *, OperandAttr> getAllOperandAttrs(Operation *op) {
  DenseMap<Value, OperandAttr> props;
  // The container attribute must be defined
  auto containerAttr = getContainerAttr<OperandAttr, Dialect>(op);
  if (!containerAttr)
    return props;

  // Map each result index to the corresponding value
  for (const NamedAttribute &attr : containerAttr.getOperandAttributes())
    props.insert(std::make_pair(op->getOperand(detail::toIdx(attr)),
                                detail::toOperandAttr<OperandAttr>(attr)));
  return props;
}

/// Sets the operand attribute of the given template type on the operand. If the
/// operand already had an attribute of this type, it is replaced.
template <typename OperandAttr, typename Dialect = handshake::HandshakeDialect>
void setOperandAttr(OpOperand &oprd, OperandAttr operandAttr) {
  Operation *owner = oprd.getOwner();
  size_t oprdIdx = oprd.getOperandNumber();
  MLIRContext *ctx = owner->getContext();

  // Create a list of attributes to store all the operand attributes of this
  // type
  SmallVector<NamedAttribute> operandAttrs;
  operandAttrs.push_back(NamedAttribute(
      StringAttr::get(ctx, std::to_string(oprdIdx)), operandAttr));

  if (auto attr = getContainerAttr<OperandAttr, Dialect>(owner)) {
    // If the operation already had operand attributes of this type, we have to
    // recreate the full map
    for (const NamedAttribute &attr : attr.getOperandAttributes()) {
      if (detail::toIdx(attr) != oprdIdx)
        operandAttrs.push_back(attr);
    }
  }
  // Set the container attribute
  auto containerAttr = OperandAttr::ContainerAttr::get(ctx, operandAttrs);
  setContainerAttr<OperandAttr>(owner, containerAttr);
}

/// Removes the operand's attribute of the given template type, if it exists.
template <typename OperandAttr, typename Dialect = handshake::HandshakeDialect>
void removeOperandAttr(OpOperand &oprd) {
  Operation *owner = oprd.getOwner();
  size_t oprdIdx = oprd.getOperandNumber();
  MLIRContext *ctx = owner->getContext();

  // Create a list of attributes to store all the operand attributes of this
  // type
  SmallVector<NamedAttribute> operandAttrs;

  // Nothing to do if the container does not exist already
  auto attr = getContainerAttr<OperandAttr, Dialect>(owner);
  if (!attr)
    return;

  // If the operation already had operand attributes of this type, we have to
  // recreate the full map
  bool foundAttribute = false;
  for (const NamedAttribute &attr : attr.getOperandAttributes()) {
    if (detail::toIdx(attr) == oprdIdx)
      // Skip over the operand attribute being removed
      foundAttribute = true;
    else
      operandAttrs.push_back(attr);
  }
  if (!foundAttribute)
    // Nothing to do if the operand didn't have the attribute
    return;

  // Set the container attribute
  auto containerAttr = OperandAttr::ContainerAttr::get(ctx, operandAttrs);
  setContainerAttr<OperandAttr>(owner, containerAttr);
}

/// Removes all operand attributes of the given template type for the
/// operation's operands.
template <typename OperandAttr, typename Dialect = handshake::HandshakeDialect>
void clearOperandAttr(Operation *op) {
  std::string name =
      detail::getDialectAttrName<Dialect, OperandAttr::ContainerAttr>();
  op->removeAttr(name);
}

} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_ATTRIBUTE_H