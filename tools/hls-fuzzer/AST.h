//===- AST.h - C AST-Node datastructures used by generators -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all datastructures that are used to represent C programs in
// memory prior to printing. It therefore serves as the IR produced by any
// generators.
//
// The common design principle among the datastructures here are:
// *) All AST-Nodes are immutable. Modifications if needed should be implemented
// as transformations producing a new AST.
// *) AST-Nodes can be used and copied by value. Container classes such as
// 'Expression' internally use reference counting which is safe to do due to
// being a tree structure.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_HLS_FUZZER_AST
#define DYNAMATIC_HLS_FUZZER_AST

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <utility>
#include <variant>
#include <vector>

/// Enable use of 'dyn_cast', 'cast' and 'isa' on 'std::variant'.
/// Primarily used to easily support these operators on classes wrapping a
/// 'std::variant'.
template <typename To, typename... Args>
struct llvm::CastInfo<To, std::variant<Args...>> {
  using From = std::variant<Args...>;

  static bool isPossible(From &f) { return std::holds_alternative<To>(f); }

  static const To &doCast(From &f) { return std::get<To>(f); }

  static const To *castFailed() { return nullptr; }

  static const To *doCastIfPossible(From &f) {
    if (!isPossible(f))
      return castFailed();
    return &doCast(f);
  }
};

template <typename To, typename... Args>
struct llvm::CastInfo<To, const std::variant<Args...>>
    : ConstStrippingForwardingCast<To, const std::variant<Args...>,
                                   CastInfo<To, std::variant<Args...>>> {};

namespace dynamatic::ast {

struct Constant;

/// Class representing a primitive type in C.
class PrimitiveType {
public:
  enum Type {
    MIN_VALUE,
    Int8 = MIN_VALUE,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Float,
    Double,
    MAX_VALUE = Double,
  };

  /// Array containing all primitive types.
  constexpr static auto ALL_PRIMITIVES = [] {
    std::array<Type, MAX_VALUE + 1> candidates{};
    std::size_t j = 0;
    for (std::size_t i = MIN_VALUE; i <= MAX_VALUE; i++)
      candidates[j++] = static_cast<Type>(i);

    return candidates;
  }();

  PrimitiveType() = default;

  /*implicit*/ PrimitiveType(Type type) : type(type) {}

  Type getType() const { return type; }

  /// Returns the number of bits the given C type occupies.
  std::size_t getBitwidth() const {
    switch (type) {
    case Int8:
    case UInt8:
      return 8;

    case Int16:
    case UInt16:
      return 16;
    case Int32:
    case UInt32:
    case Float:
      return 32;
    case Double:
      return 64;
    }
    llvm_unreachable("all enum cases handled");
  }

  /// Returns true if this primitive type is an integer type.
  bool isInteger() const {
    switch (type) {
    case Float:
    case Double:
      return false;
    default:
      return true;
    }
  }

  /// Returns true if this primitive type can directly represent signed values.
  bool isSigned() const {
    switch (type) {
    case UInt8:
    case UInt16:
    case UInt32:
      return false;
    default:
      return true;
    }
  }

  /// Returns the smallest possible value for the given primitive type.
  Constant getMinValue() const;

  /// Returns the largest possible value for the given primitive type.
  Constant getMaxValue() const;

  friend bool operator==(PrimitiveType lhs, PrimitiveType rhs) {
    return lhs.type == rhs.type;
  }

  friend bool operator!=(PrimitiveType lhs, PrimitiveType rhs) {
    return !(lhs == rhs);
  }

private:
  Type type{};
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const PrimitiveType &primitive);

/// Container class representing a scalar data type in C.
/// Use LLVM's casting utilities for downcasting.
/// Lifetimes of types are handled internally via reference counting and do not
/// need to be managed by users.
class ScalarType {
  // TODO: Add Bitwidth type.
  using Variant = std::variant<PrimitiveType>;

public:
  ScalarType() = default;

  template <class T, std::enable_if_t<std::is_constructible_v<Variant, T> &&
                                      !std::is_same_v<std::decay_t<Variant>, T>>
                         * = nullptr>
  /*implicit*/ ScalarType(T &&arg)
      : datatype(std::make_shared<Variant>(std::forward<T>(arg))) {}

  friend bool operator==(const ScalarType &lhs, const ScalarType &rhs);

  friend bool operator!=(const ScalarType &lhs, const ScalarType &rhs) {
    return !(lhs == rhs);
  }

  /// Returns the number of bits required to represent the given type.
  std::size_t getBitwidth() const;

  /// Returns true if the given type can directly represent signed values.
  bool isSigned() const;

  bool isInteger() const;

  template <typename From>
  friend struct llvm::simplify_type;

  using SubElements = std::tuple<>;

private:
  std::shared_ptr<const Variant> datatype;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const ScalarType &scalarType);

/// AST-node representing literals in C.
/// This is a variant between all possible integer and floating point types.
/// The type of the expression corresponds directly to the type of the value.
struct Constant {
  using Variant =
      std::variant<std::int8_t, std::uint8_t, std::int16_t, std::uint16_t,
                   std::int32_t, std::uint32_t, float, double>;
  Variant value;

  /// Returns the type of this expression.
  PrimitiveType getType() const {
    return llvm::TypeSwitch<Variant, PrimitiveType>(value)
        .Case(
            [](const int8_t *) -> PrimitiveType { return PrimitiveType::Int8; })
        .Case([](const uint8_t *) -> PrimitiveType {
          return PrimitiveType::UInt8;
        })
        .Case([](const int16_t *) -> PrimitiveType {
          return PrimitiveType::Int16;
        })
        .Case([](const uint16_t *) -> PrimitiveType {
          return PrimitiveType::UInt16;
        })
        .Case([](const int32_t *) -> PrimitiveType {
          return PrimitiveType::Int32;
        })
        .Case([](const uint32_t *) -> PrimitiveType {
          return PrimitiveType::UInt32;
        })
        .Case(
            [](const float *) -> PrimitiveType { return PrimitiveType::Float; })
        .Case([](const double *) -> PrimitiveType {
          return PrimitiveType::Double;
        });
  }

  using SubElements = std::tuple<>;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Constant &constant);

class ScalarParameter;

/// AST-node representing a reference to a variable in C.
struct Variable {
  const ScalarType datatype;
  const std::string name;

  const ScalarType &getType() const { return datatype; }

  using SubElements = std::tuple<ScalarParameter>;
  constexpr static std::size_t PARAMETER = 0;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Variable &variable);

class BinaryExpression;
class CastExpression;
class UnaryExpression;
class ConditionalExpression;
class ArrayReadExpression;

/// Super-type of all AST-nodes representing expressions.
/// Lifetimes of objects are handled internally via reference counting and does
/// not need to be handled by users.
class Expression {
  using Variant =
      std::variant<Constant, Variable, BinaryExpression, CastExpression,
                   UnaryExpression, ConditionalExpression, ArrayReadExpression>;

public:
  Expression() = default;

  /// Construct an 'Expression' implicitly from any concrete AST-node.
  template <class T,
            std::enable_if_t<std::conjunction_v<
                std::negation<std::is_same<std::decay_t<T>, Expression>>,
                std::is_constructible<Variant, T>>> * = nullptr>
  /*implicit*/ Expression(T &&arg)
      : expression(std::make_shared<Variant>(std::forward<T>(arg))) {}

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const Expression &expression);

  /// Returns the type of the given expression.
  ScalarType getType() const;

private:
  std::shared_ptr<const Variant> expression;
};

/// AST-Node representing all binary expressions in C.
class BinaryExpression {
public:
  enum Op {
    MIN_VALUE,
    BitAnd = MIN_VALUE,
    BitOr,
    BitXor,
    ShiftLeft,
    ShiftRight,
    Plus,
    Minus,
    Mul,
    // Division,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    Equal,
    NotEqual,
    MAX_VALUE = NotEqual,
  };

  BinaryExpression(Expression lhs, Op op, Expression rhs)
      : lhs(std::move(lhs)), op(op), rhs(std::move(rhs)) {}

  const Expression &getLhs() const { return lhs; }

  Op getOp() const { return op; }

  const Expression &getRhs() const { return rhs; }

  /// Returns the type of the given expression.
  ScalarType getType() const;

  /// Returns true if a value of type 'datatype' is allowed to be used as an
  /// operand for 'op'.
  static bool isLegalOperandType(Op op, const ScalarType &datatype);

  using SubElements = std::tuple<Expression, Expression>;

  constexpr static std::size_t LHS = 0;
  constexpr static std::size_t RHS = 1;

private:
  Expression lhs;
  Op op;
  Expression rhs;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const BinaryExpression &binOp);

/// AST-Node representing a cast expression in C.
class CastExpression {
public:
  CastExpression(ScalarType targetType, Expression expression)
      : targetType(std::move(targetType)), expression(std::move(expression)) {}

  const Expression &getExpression() const { return expression; }

  /// Returns the type of this expression, i.e., the type being cast to.
  const ScalarType &getType() const { return targetType; }

  using SubElements = std::tuple<ScalarType, Expression>;

  constexpr static std::size_t TARGET_TYPE = 0;
  constexpr static std::size_t OPERAND = 1;

private:
  ScalarType targetType;
  Expression expression;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const CastExpression &castExpression);

/// AST-Node representing all unary expressions in C.
class UnaryExpression {
public:
  enum Op {
    MIN_VALUE,
    BitwiseNot = MIN_VALUE,
    BoolNot,
    Minus,
    MAX_VALUE = Minus,
  };

  UnaryExpression(Op op, Expression expression)
      : op(op), expression(std::move(expression)) {}

  Op getOp() const { return op; }

  const Expression &getExpression() const { return expression; }

  ScalarType getType() const;

  static bool isLegalOperandType(Op op, const ScalarType &type);

  using SubElements = std::tuple<Expression>;

private:
  Op op;
  Expression expression;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const UnaryExpression &unaryExpression);

/// AST-Node representing a conditional expression (cond ? true : false) in C.
class ConditionalExpression {
public:
  ConditionalExpression(Expression condition, Expression trueVal,
                        Expression falseVal)
      : condition(std::move(condition)), trueVal(std::move(trueVal)),
        falseVal(std::move(falseVal)) {}

  const Expression &getCondition() const { return condition; }

  const Expression &getTrueVal() const { return trueVal; }

  const Expression &getFalseVal() const { return falseVal; }

  ScalarType getType() const;

  using SubElements = std::tuple<Expression, Expression, Expression>;
  constexpr static std::size_t CONDITION = 0;
  constexpr static std::size_t TRUE_VAL = 1;
  constexpr static std::size_t FALSE_VAL = 2;

private:
  Expression condition;
  Expression trueVal;
  Expression falseVal;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const ConditionalExpression &ternaryExpression);

class ArrayParameter;

/// Expression representing reading and indexing into an array.
/// Only array parameters and one-dimensional arrays are currently supported.
class ArrayReadExpression {
public:
  /// Constructs an array-read expression from an array element type, the name
  /// of the array parameter and the indexing expression.
  ArrayReadExpression(ScalarType elementType, std::string arrayParameter,
                      Expression index)
      : dataType(std::move(elementType)),
        arrayParameter(std::move(arrayParameter)), index(std::move(index)) {}

  /// Returns the name of the array parameters.
  llvm::StringRef getArrayParameter() const { return arrayParameter; }

  /// Returns the indexing expression.
  const Expression &getIndex() const { return index; }

  /// Returns the result-type of the expression, which is equivalent to the
  /// element type of the array.
  const ScalarType &getType() const { return dataType; }

  using SubElements = std::tuple<ArrayParameter, Expression>;

  constexpr static std::size_t ARRAY_PARAMETER = 0;
  constexpr static std::size_t INDEX = 1;

private:
  ScalarType dataType;
  std::string arrayParameter;
  Expression index;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const ArrayReadExpression &arrayReadExpression);

/// AST-Node representing a return statement in C.
class ReturnStatement {
public:
  explicit ReturnStatement(Expression returnValue)
      : returnValue(std::move(returnValue)) {}

  const Expression &getReturnValue() const { return returnValue; }

  using SubElements = std::tuple<Expression>;
  constexpr static std::size_t RETURN_VALUE = 0;

private:
  Expression returnValue;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const ReturnStatement &statement);

/// AST-Node representing an assignment to an element of an array.
/// Note that we model this as a top-level statement despite this being an
/// expression in C.
class ArrayAssignmentStatement {
public:
  ArrayAssignmentStatement(std::string arrayParameter,
                           Expression indexingExpression,
                           Expression valueExpression)
      : arrayParameter(std::move(arrayParameter)),
        indexingExpression(std::move(indexingExpression)),
        valueExpression(std::move(valueExpression)) {}

  llvm::StringRef getArrayParameter() const { return arrayParameter; }

  const Expression &getIndexingExpression() const { return indexingExpression; }

  /// Returns the value that will be assigned to the element.
  const Expression &getValueExpression() const { return valueExpression; }

  using SubElements = std::tuple<ArrayParameter, Expression, Expression>;
  constexpr static std::size_t ARRAY = 0;
  constexpr static std::size_t INDEX = 1;
  constexpr static std::size_t VALUE = 2;

private:
  std::string arrayParameter;
  Expression indexingExpression;
  Expression valueExpression;
};

llvm::raw_ostream &
operator<<(llvm::raw_ostream &os,
           const ArrayAssignmentStatement &arrayAssignmentStatement);

class Statement {
  using Variant = std::variant<ArrayAssignmentStatement>;

public:
  Statement() = default;

  template <class T, std::enable_if_t<std::is_constructible_v<Variant, T> &&
                                      !std::is_same_v<std::decay_t<Variant>, T>>
                         * = nullptr>
  /*implicit*/ Statement(T &&arg)
      : statement(std::make_shared<Variant>(std::forward<T>(arg))) {}

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const Statement &statement);

private:
  std::shared_ptr<const Variant> statement;
};

/// Class representing a list of statements in a body.
class StatementList {
public:
  StatementList() = default;

  explicit StatementList(std::vector<Statement> statements)
      : statements(std::move(statements)) {}

  /// Returns the number of statements.
  std::size_t size() const { return statements.size(); }

  const Statement &operator[](std::size_t index) const {
    return statements[index];
  }

  auto begin() { return statements.begin(); }

  auto begin() const { return statements.begin(); }

  auto end() { return statements.end(); }

  auto end() const { return statements.end(); }

  std::vector<Statement> takeVector() { return std::move(statements); }

  // Recursive statement list representation.
  // The definition is left recursive, meaning the statement is always the tail
  // statement after the list.
  using SubElements = std::tuple<StatementList, Statement>;

private:
  std::vector<Statement> statements;
};

/// AST-Node representing a scalar function parameter in C.
class ScalarParameter {
public:
  ScalarParameter() = default;

  ScalarParameter(ScalarType dataType, std::string name)
      : dataType(std::move(dataType)), name(std::move(name)) {}

  llvm::StringRef getName() const { return name; }

  const ScalarType &getDataType() const { return dataType; }

  using SubElements = std::tuple<ScalarType>;
  constexpr static std::size_t DATATYPE = 0;

private:
  ScalarType dataType;
  std::string name;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const ScalarParameter &parameter);

/// Variant of 'ScalarParameter' used to represent an existing 'ScalarParameter'
/// in the generator.
/// Unlike 'ScalarParameter' it doesn't have any subelements and is a terminal
/// similar to 'Constant'.
class ExistingScalarParameter : public ScalarParameter {
public:
  /*implicit*/ ExistingScalarParameter(const ScalarParameter &scalarParameter)
      : ScalarParameter(scalarParameter) {}

  /*implicit*/ ExistingScalarParameter(ScalarParameter &&scalarParameter)
      : ScalarParameter(std::move(scalarParameter)) {}

  using SubElements = std::tuple<>;
};

/// AST-Node representing an array function parameter in C.
class ArrayParameter {
public:
  ArrayParameter() = default;

  ArrayParameter(ScalarType elementType, std::string name,
                 std::size_t dimension)
      : dataType(std::move(elementType)), name(std::move(name)),
        dimension(dimension) {
    assert(dimension > 0);
  }

  llvm::StringRef getName() const { return name; }

  const ScalarType &getElementType() const { return dataType; }

  std::size_t getDimension() const { return dimension; }

  using SubElements = std::tuple<ScalarType>;
  constexpr static std::size_t ELEMENT_TYPE = 0;

private:
  ScalarType dataType;
  std::string name;
  std::size_t dimension;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const ArrayParameter &parameter);

/// Variant of 'ArrayParameter' used to represent an existing 'ArrayParameter'
/// in the generator.
/// Unlike 'ArrayParameter' it doesn't have any subelements and is a terminal
/// similar to 'Constant'.
class ExistingArrayParameter : public ArrayParameter {
public:
  /*implicit*/ ExistingArrayParameter(const ArrayParameter &arrayParameter)
      : ArrayParameter(arrayParameter) {}

  /*implicit*/ ExistingArrayParameter(ArrayParameter &&arrayParameter)
      : ArrayParameter(std::move(arrayParameter)) {}

  using SubElements = std::tuple<>;
};

/// Tag type representing the 'void' type from C.
struct VoidType {
  friend bool operator==(const VoidType &lhs, const VoidType &rhs) {
    return true;
  }

  friend bool operator!=(const VoidType &lhs, const VoidType &rhs) {
    return !(lhs == rhs);
  }
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const VoidType &) {
  return os << "void";
}

class ReturnType {
  using Variant = std::variant<VoidType, ScalarType>;

public:
  ReturnType() = default;

  template <class T, std::enable_if_t<std::is_constructible_v<Variant, T> &&
                                      !std::is_same_v<std::decay_t<Variant>, T>>
                         * = nullptr>
  /*implicit*/ ReturnType(T &&arg) : variant(std::forward<T>(arg)) {}

  friend bool operator==(const ReturnType &lhs, const ReturnType &rhs) {
    return lhs.variant == rhs.variant;
  }

  friend bool operator!=(const ReturnType &lhs, const ReturnType &rhs) {
    return !(lhs == rhs);
  }

  template <typename From>
  friend struct llvm::simplify_type;

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const ReturnType &returnType);

  using SubElements = std::tuple<>;

private:
  Variant variant;
};

/// AST-Node representing a function in C.
/// Functions are currently limited to just a return statement.
struct Function {
  /// Void or a return type.
  const ReturnType returnType;
  const std::string name;
  const std::vector<ScalarParameter> scalarParameters;
  const std::vector<ArrayParameter> arrayParameters;

  const std::vector<Statement> statements;
  /// The return statement at the end of a function iff it does not have a void
  /// return type.
  const std::optional<ReturnStatement> returnStatement;

  using SubElements = std::tuple<ReturnType, StatementList, ReturnStatement>;
  constexpr static std::size_t RETURN_TYPE = 0;
  constexpr static std::size_t STATEMENTS = 1;
  constexpr static std::size_t RETURN_STATEMENT = 2;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Function &function);

template <class... Args>
llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const std::variant<Args...> &variant) {
  return std::visit([&](auto &&value) -> decltype(auto) { return os << value; },
                    variant);
}

inline bool operator==(const ScalarType &lhs, const ScalarType &rhs) {
  return *lhs.datatype == *rhs.datatype;
}

inline std::size_t ScalarType::getBitwidth() const {
  return std::visit(
      [&](auto &&value) -> std::size_t { return value.getBitwidth(); },
      *datatype);
}

inline bool ScalarType::isSigned() const {
  return std::visit([&](auto &&value) -> bool { return value.isSigned(); },
                    *datatype);
}

inline bool ScalarType::isInteger() const {
  return std::visit([&](auto &&value) -> bool { return value.isInteger(); },
                    *datatype);
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const Expression &expression) {
  return os << *expression.expression;
}

inline ScalarType Expression::getType() const {
  return std::visit([&](auto &&value) -> ScalarType { return value.getType(); },
                    *expression);
}

} // namespace dynamatic::ast

namespace dynamatic {
/// Returns a range of all enum values from 'MIN_VALUE' to incl 'MAX_VALUE'.
template <typename EnumT>
auto enumRange() {
  return llvm::map_range(
      llvm::iota_range<std::size_t>(static_cast<std::size_t>(EnumT::MIN_VALUE),
                                    static_cast<std::size_t>(EnumT::MAX_VALUE),
                                    /*Inclusive=*/true),
      [](std::size_t i) { return static_cast<EnumT>(i); });
}
} // namespace dynamatic

// Enable 'dyn_cast' and friends on 'ScalarType' by delegating to 'dyn_cast' on
// the variant.
// E.g. 'ast::PrimitiveType* prim = dyn_cast<ast::PrimitiveType>(scalarType);'
template <>
struct llvm::simplify_type<dynamatic::ast::ScalarType> {
  using SimpleType = const dynamatic::ast::ScalarType::Variant;

  static SimpleType &
  getSimplifiedValue(const dynamatic::ast::ScalarType &datatype) {
    return *datatype.datatype;
  }
};

template <>
struct llvm::simplify_type<dynamatic::ast::ReturnType> {
  using SimpleType = const dynamatic::ast::ReturnType::Variant;

  static SimpleType &
  getSimplifiedValue(const dynamatic::ast::ReturnType &datatype) {
    return datatype.variant;
  }
};

#endif
