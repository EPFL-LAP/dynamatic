//===- PropertyExpr.h - Expressions of formal properties --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the expression language in which formal properties are stated: a
// boolean, integer, and CTL formula over the state and signals of a circuit,
// which can be serialized to JSON and printed in SMV syntax.
//
// As in the C AST of hls-fuzzer, each kind of node is a class holding only its
// own fields, and 'PropertyExpr' is a container holding any of them. Nodes are
// immutable and are shared between expressions, so copying an expression is
// cheap.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_SUPPORT_PROPERTYEXPR_H
#define EXPERIMENTAL_SUPPORT_PROPERTYEXPR_H

#include "dynamatic/Dialect/Handshake/HandshakeOpInternalStateNamer.h"
#include "dynamatic/Support/LLVM.h"
#include "llvm/Support/JSON.h"
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace dynamatic {

class State;
class Signal;
class Constant;
class UnaryExpr;
class BinaryExpr;
class NaryExpr;

/// Container of any node of a property expression.
class PropertyExpr {
public:
  using Variant =
      std::variant<State, Signal, Constant, UnaryExpr, BinaryExpr, NaryExpr>;

  PropertyExpr() = default;

  /// Constructs an expression implicitly from any node. This is equivalent to
  /// writing out one plain constructor per node, i.e., PropertyExpr(State
  /// node), PropertyExpr(Signal node), ..., PropertyExpr(NaryExpr node). The
  /// template only exists for the node types (`is_constructible<Variant, T>`),
  /// and not for PropertyExpr itself, so that copying an expression still uses
  /// the copy constructor.
  template <class T,
            std::enable_if_t<std::conjunction_v<
                std::negation<std::is_same<std::decay_t<T>, PropertyExpr>>,
                std::is_constructible<Variant, T>>> * = nullptr>
  /*implicit*/ PropertyExpr(T &&node)
      : node(std::make_shared<const Variant>(std::forward<T>(node))) {}

  /// The node held by the expression.
  const Variant &getNode() const;

  /// Whether the expression contains a temporal operator, i.e., whether it can
  /// only be checked as a CTL formula rather than as an invariant.
  bool isTemporal() const;

  /// Calls `fn` on this expression and on all its subexpressions, in
  /// pre-order.
  void walk(llvm::function_ref<void(const PropertyExpr &)> fn) const;

  /// Prints the expression in SMV syntax. Every operation is parenthesized, so
  /// that the printed formula does not depend on the operator precedence of
  /// SMV.
  std::string toSMV() const;

private:
  std::shared_ptr<const Variant> node;
};

llvm::json::Value toJSON(const PropertyExpr &expr);
bool fromJSON(const llvm::json::Value &value, PropertyExpr &expr,
              llvm::json::Path path);

/// The internal state of a unit (e.g., whether a buffer slot is full).
class State {
public:
  /// If `constraint` is set, the state only counts if it holds a token of that
  /// value (see `InternalStateNamer::tryConstrain`).
  State(std::shared_ptr<handshake::InternalStateNamer> namer,
        std::optional<int32_t> constraint = std::nullopt)
      : namer(std::move(namer)), constraint(constraint) {
    assert(this->namer && "state without a namer");
  }

  /// Copies `namer`.
  template <typename Namer, typename = std::enable_if_t<std::is_base_of_v<
                                handshake::InternalStateNamer, Namer>>>
  State(const Namer &namer, std::optional<int32_t> constraint = std::nullopt)
      : State(std::shared_ptr<handshake::InternalStateNamer>(
                  std::make_shared<Namer>(namer)),
              constraint) {}

  const std::shared_ptr<handshake::InternalStateNamer> &getNamer() const {
    return namer;
  }
  std::optional<int32_t> getConstraint() const { return constraint; }

private:
  std::shared_ptr<handshake::InternalStateNamer> namer;
  std::optional<int32_t> constraint;
};

/// The valid or ready signal of a port of a unit (e.g., `fork0.outs_0_valid`).
class Signal {
public:
  enum Kind { Valid, Ready };
  /// Whether the port is an operand or a result of the unit.
  enum PortSide { Operand, Result };

  /// `portIndex` is the index of the operand or result, depending on `side`,
  /// and `portName` its name (see `NamedIOInterface`).
  Signal(Kind kind, StringRef opName, PortSide side, unsigned portIndex,
         StringRef portName)
      : kind(kind), opName(opName.str()), side(side), portIndex(portIndex),
        portName(portName.str()) {}

  Kind getKind() const { return kind; }
  StringRef getOpName() const { return opName; }
  PortSide getSide() const { return side; }
  unsigned getPortIndex() const { return portIndex; }
  StringRef getPortName() const { return portName; }

private:
  Kind kind;
  std::string opName;
  PortSide side;
  unsigned portIndex;
  std::string portName;
};

/// An integer or boolean constant.
class Constant {
public:
  using Value = std::variant<bool, int64_t>;

  explicit Constant(Value value) : value(value) {}
  // Named constructors, as a literal like `0` converts to either alternative
  static Constant integer(int64_t value) { return Constant(Value(value)); }
  static Constant boolean(bool value) { return Constant(Value(value)); }

  const Value &getValue() const { return value; }

private:
  Value value;
};

class UnaryExpr {
public:
  enum Op { Not, ToInt, AG, AF, AX, EG, EF, EX };

  UnaryExpr(Op op, PropertyExpr operand)
      : op(op), operand(std::move(operand)) {}

  Op getOp() const { return op; }
  const PropertyExpr &getOperand() const { return operand; }

  /// Whether `op` is a CTL operator.
  static bool isTemporal(Op op) { return op != Not && op != ToInt; }

private:
  Op op;
  PropertyExpr operand;
};

class BinaryExpr {
public:
  enum Op { Implies, Iff, Sub, Mul, Eq, Lt, Le, Gt, Ge };

  BinaryExpr(PropertyExpr lhs, Op op, PropertyExpr rhs)
      : lhs(std::move(lhs)), op(op), rhs(std::move(rhs)) {}

  Op getOp() const { return op; }
  const PropertyExpr &getLhs() const { return lhs; }
  const PropertyExpr &getRhs() const { return rhs; }

private:
  PropertyExpr lhs;
  Op op;
  PropertyExpr rhs;
};

/// An operation over one or more operands.
class NaryExpr {
public:
  enum Op { And, Or, Add, Count };

  NaryExpr(Op op, std::vector<PropertyExpr> operands)
      : op(op), operands(std::move(operands)) {
    // SMV's count takes at least one argument, and the other operations are
    // built with `conjunction`, `disjunction`, and `sum` when there may be none
    assert(!this->operands.empty() && "operation without operands");
  }

  Op getOp() const { return op; }
  ArrayRef<PropertyExpr> getOperands() const { return operands; }

private:
  Op op;
  std::vector<PropertyExpr> operands;
};

/// Conjunction of `operands`: TRUE if there are none, and the operand itself
/// if there is one.
PropertyExpr conjunction(std::vector<PropertyExpr> operands);
/// Disjunction of `operands`: FALSE if there are none, and the operand itself
/// if there is one.
PropertyExpr disjunction(std::vector<PropertyExpr> operands);
/// Sum of `operands`: 0 if there are none, and the operand itself if there is
/// one.
PropertyExpr sum(std::vector<PropertyExpr> operands);

// Defined after the nodes, as accessing the variant requires them to be
// complete
inline const PropertyExpr::Variant &PropertyExpr::getNode() const {
  return *node;
}

inline PropertyExpr operator!(PropertyExpr operand) {
  return UnaryExpr(UnaryExpr::Not, std::move(operand));
}

inline PropertyExpr implies(PropertyExpr lhs, PropertyExpr rhs) {
  return BinaryExpr(std::move(lhs), BinaryExpr::Implies, std::move(rhs));
}

} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_PROPERTYEXPR_H
