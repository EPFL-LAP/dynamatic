#include "AST.h"

#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cmath>

using namespace dynamatic;

ast::Constant ast::PrimitiveType::getMinValue() const {
  switch (type) {
  case Int8:
    return {std::numeric_limits<int8_t>::min()};
  case UInt8:
    return {std::numeric_limits<uint8_t>::min()};
  case Int16:
    return {std::numeric_limits<int16_t>::min()};
  case UInt16:
    return {std::numeric_limits<uint16_t>::min()};
  case Int32:
    return {std::numeric_limits<int32_t>::min()};
  case UInt32:
    return {std::numeric_limits<uint32_t>::min()};
  case Float:
    return {std::numeric_limits<float>::min()};
  case Double:
    return {std::numeric_limits<double>::min()};
  }
  llvm_unreachable("all enum cases handled");
}

ast::Constant ast::PrimitiveType::getMaxValue() const {
  switch (type) {
  case Int8:
    return {std::numeric_limits<int8_t>::max()};
  case UInt8:
    return {std::numeric_limits<uint8_t>::max()};
  case Int16:
    return {std::numeric_limits<int16_t>::max()};
  case UInt16:
    return {std::numeric_limits<uint16_t>::max()};
  case Int32:
    return {std::numeric_limits<int32_t>::max()};
  case UInt32:
    return {std::numeric_limits<uint32_t>::max()};
  case Float:
    return {std::numeric_limits<float>::max()};
  case Double:
    return {std::numeric_limits<double>::max()};
  }
  llvm_unreachable("all enum cases handled");
}

llvm::raw_ostream &ast::operator<<(llvm::raw_ostream &os,
                                   const PrimitiveType &primitive) {
  switch (primitive.getType()) {
  case PrimitiveType::Int8:
    return os << "int8_t";
  case PrimitiveType::UInt8:
    return os << "uint8_t";
  case PrimitiveType::Int16:
    return os << "int16_t";
  case PrimitiveType::UInt16:
    return os << "uint16_t";
  case PrimitiveType::Int32:
    return os << "int32_t";
  case PrimitiveType::UInt32:
    return os << "uint32_t";
  case PrimitiveType::Float:
    return os << "float";
  case PrimitiveType::Double:
    return os << "double";
  }
  std::abort();
}

llvm::raw_ostream &ast::operator<<(llvm::raw_ostream &os,
                                   const ScalarType &scalarType) {
  llvm::TypeSwitch<ScalarType>(scalarType)
      .Case([&](const PrimitiveType *primitive) { os << *primitive; });
  return os;
}

llvm::raw_ostream &ast::operator<<(llvm::raw_ostream &os,
                                   const ScalarParameter &parameter) {

  return os << parameter.getDataType() << " " << parameter.getName();
}

llvm::raw_ostream &ast::operator<<(llvm::raw_ostream &os,
                                   const Constant &constant) {
  llvm::TypeSwitch<Constant::Variant>(constant.value)
      .Case([&](const int32_t *value) { os << *value; })
      .Case([&](const uint32_t *value) { os << *value << 'u'; })
      .Case([&](const int8_t *value) {
        os << "(int8_t)(" << static_cast<int32_t>(*value) << ")";
      })
      .Case([&](const uint8_t *value) {
        os << "(uint8_t)(" << static_cast<int32_t>(*value) << ")";
      })
      .Case([&](const int16_t *value) { os << "(int16_t)(" << *value << ")"; })
      .Case(
          [&](const uint16_t *value) { os << "(uint16_t)(" << *value << ")"; })
      .Case<float, double>([&](const auto *value) {
        using T = std::decay_t<decltype(*value)>;
        if (std::isinf(*value)) {
          if constexpr (std::is_same_v<T, double>) {
            os << "HUGE_VAL";
            return;
          }
          os << "HUGE_VALF";
          return;
        }
        if (std::isnan(*value)) {
          os << "NAN";
          return;
        }
        os << *value;
      });
  return os;
}

llvm::raw_ostream &ast::operator<<(llvm::raw_ostream &os,
                                   const Variable &variable) {
  return os << variable.name;
}

/// Performs "integer promotion" on 'type'.
/// Integer promotion in C is performed on an integer type smaller than 'int'
/// by upcasting it to 'int'.
/// For reference, see 6.3.1.1 in the C23 standard.
/// Note we assume 'int' to be equal to 'int32_t'.
static void integerPromotion(ast::ScalarType &type) {
  auto *primType = llvm::dyn_cast<ast::PrimitiveType>(type);
  if (!primType)
    return;

  switch (primType->getType()) {
  case ast::PrimitiveType::UInt8:
  case ast::PrimitiveType::Int8:
  case ast::PrimitiveType::Int16:
  case ast::PrimitiveType::UInt16:
    type = ast::PrimitiveType::Int32;
    break;
  default:
    break;
  }
}

/// Performs what the C23 standard calls "the usual arithmetic conversion".
/// This performs the required upcasting in any binary operation to a common
/// type that the operation is then performed on.
/// For reference, see 6.3.1.8 in the C23 standard.
static ast::ScalarType usualArithmeticConversion(ast::ScalarType lhs,
                                                 ast::ScalarType rhs) {
  if (lhs == ast::PrimitiveType::Double || rhs == ast::PrimitiveType::Double)
    return ast::PrimitiveType::Double;

  if (lhs == ast::PrimitiveType::Float || rhs == ast::PrimitiveType::Float)
    return ast::PrimitiveType::Float;

  // Dealing with only integer operations at this point.
  integerPromotion(lhs);
  integerPromotion(rhs);

  // The operation is performed on the larger integer type at this point, where
  // larger means higher bitwidth or the unsigned integer type if the bitwidth
  // is equal.
  // Note: We do not support int64_t and uint64_t, meaning we use either
  // uint32_t or int32_t here.
  if (lhs == ast::PrimitiveType::UInt32 || rhs == ast::PrimitiveType::UInt32)
    return ast::PrimitiveType::UInt32;

  assert(lhs == rhs);
  return lhs;
}

ast::ScalarType ast::BinaryExpression::getType() const {
  switch (op) {
  case BitAnd:
  case BitOr:
  case BitXor:
  case Plus:
  case Minus:
  case Mul:
    // case Division:
    return usualArithmeticConversion(lhs.getType(), rhs.getType());
  case ShiftRight:
  case ShiftLeft: {
    ScalarType lhsType = lhs.getType();
    integerPromotion(lhsType);
    return lhsType;
  }
  case Greater:
  case GreaterEqual:
  case Less:
  case LessEqual:
  case Equal:
  case NotEqual:
    return PrimitiveType::Int32;
  }
  llvm_unreachable("all enum cases handled");
}

bool ast::BinaryExpression::isLegalOperandType(Op op,
                                               const ScalarType &datatype) {
  switch (op) {
  case BitAnd:
  case BitOr:
  case BitXor:
  case ShiftLeft:
  case ShiftRight: {
    auto *prim = llvm::dyn_cast<PrimitiveType>(datatype);
    if (!prim)
      return false;
    return prim->isInteger();
  }
  case Plus:
  case Minus:
  case Mul:
  // case Division:
  case Greater:
  case GreaterEqual:
  case Less:
  case LessEqual:
  case Equal:
  case NotEqual:
    return true;
  }
  llvm_unreachable("all enum cases handled");
}

llvm::raw_ostream &ast::operator<<(llvm::raw_ostream &os,
                                   const BinaryExpression &binOp) {
  os << "(" << binOp.getLhs() << " ";
  switch (binOp.getOp()) {
  case BinaryExpression::BitAnd:
    os << '&';
    break;
  case BinaryExpression::BitOr:
    os << '|';
    break;
  case BinaryExpression::BitXor:
    os << '^';
    break;
  case BinaryExpression::ShiftLeft:
    os << "<<";
    break;
  case BinaryExpression::ShiftRight:
    os << ">>";
    break;
  case BinaryExpression::Plus:
    os << '+';
    break;
  case BinaryExpression::Minus:
    os << '-';
    break;
  case BinaryExpression::Mul:
    os << '*';
    break;
  // case BinaryExpression::Division:
  //   os << '/';
  //   break;
  case BinaryExpression::Greater:
    os << '>';
    break;
  case BinaryExpression::GreaterEqual:
    os << ">=";
    break;
  case BinaryExpression::Less:
    os << '<';
    break;
  case BinaryExpression::LessEqual:
    os << "<=";
    break;
  case BinaryExpression::Equal:
    os << "==";
    break;
  case BinaryExpression::NotEqual:
    os << "!=";
    break;
  }
  os << " " << binOp.getRhs() << ")";
  return os;
}

llvm::raw_ostream &ast::operator<<(llvm::raw_ostream &os,
                                   const CastExpression &castExpression) {
  return os << "(" << castExpression.getType() << ")("
            << castExpression.getExpression() << ")";
}

ast::ScalarType ast::UnaryExpression::getType() const {
  switch (op) {
  case BitwiseNot:
  case Minus: {
    ScalarType operandType = expression.getType();
    integerPromotion(operandType);
    return operandType;
  }
  case BoolNot:
    return PrimitiveType::Int32;
  }
  llvm_unreachable("all enum cases handled");
}

llvm::raw_ostream &ast::operator<<(llvm::raw_ostream &os,
                                   const UnaryExpression &unaryExpression) {
  os << '(';
  switch (unaryExpression.getOp()) {
  case UnaryExpression::BitwiseNot:
    os << '~';
    break;
  case UnaryExpression::Minus:
    os << '-';
    break;
  case UnaryExpression::BoolNot:
    os << '!';
    break;
  }
  return os << unaryExpression.getExpression() << ')';
}

ast::ScalarType ast::ConditionalExpression::getType() const {
  return usualArithmeticConversion(trueVal.getType(), falseVal.getType());
}

llvm::raw_ostream &
ast::operator<<(llvm::raw_ostream &os,
                const ConditionalExpression &ternaryExpression) {
  return os << "(" << ternaryExpression.getCondition() << " ? "
            << ternaryExpression.getTrueVal() << " : "
            << ternaryExpression.getFalseVal() << ")";
}

llvm::raw_ostream &
ast::operator<<(llvm::raw_ostream &os,
                const ArrayReadExpression &arrayReadExpression) {
  return os << arrayReadExpression.getArrayParameter() << '['
            << arrayReadExpression.getIndex() << ']';
}

llvm::raw_ostream &ast::operator<<(llvm::raw_ostream &os,
                                   const ReturnStatement &statement) {
  return os << "return " << statement.getReturnValue() << ";";
}

llvm::raw_ostream &
ast::operator<<(llvm::raw_ostream &os,
                const ArrayAssignmentStatement &arrayAssignmentStatement) {
  return os << arrayAssignmentStatement.getArrayParameter() << '['
            << arrayAssignmentStatement.getIndexingExpression()
            << "] = " << arrayAssignmentStatement.getValueExpression() << ';';
}

llvm::raw_ostream &ast::operator<<(llvm::raw_ostream &os,
                                   const Statement &statement) {
  return os << *statement.statement;
}

llvm::raw_ostream &ast::operator<<(llvm::raw_ostream &os,
                                   const ArrayParameter &parameter) {
  return os << parameter.getElementType() << ' ' << parameter.getName() << '['
            << parameter.getDimension() << ']';
}

llvm::raw_ostream &ast::operator<<(llvm::raw_ostream &os,
                                   const Function &function) {
  os << function.returnType << ' ' << function.name << '(';
  llvm::interleaveComma(function.scalarParameters, os);
  if (!function.scalarParameters.empty() && !function.arrayParameters.empty())
    os << ", ";
  llvm::interleaveComma(function.arrayParameters, os);
  os << ") {\n";

  mlir::raw_indented_ostream indentedOstream(os);
  indentedOstream.indent();
  for (auto &iter : function.statements)
    indentedOstream << iter;
  if (function.returnStatement)
    indentedOstream << *function.returnStatement;

  os << "\n}\n";
  return os;
}
