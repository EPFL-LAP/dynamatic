#include "DynamaticTypeSystem.h"

auto dynamatic::gen::DynamaticTypeSystem::checkScalarType(
    const ast::ScalarType &scalarType, DynamaticTypingContext context)
    -> std::optional<ConclusionOf<ast::ScalarType>> {
  switch (context.constraint) {
  case DynamaticTypingContext::FloatRequired:
    if (scalarType != ast::PrimitiveType::Float &&
        scalarType != ast::PrimitiveType::Double)
      return std::nullopt;
    return ConclusionOf<ast::ScalarType>{};

  case DynamaticTypingContext::IntegerRequired:
    if (scalarType == ast::PrimitiveType::Float ||
        scalarType == ast::PrimitiveType::Double)
      return std::nullopt;
    return ConclusionOf<ast::ScalarType>{};
  }
  llvm_unreachable("all enum cases handled");
}

auto dynamatic::gen::DynamaticTypeSystem::checkBinaryExpression(
    ast::BinaryExpression::Op op, DynamaticTypingContext context)
    -> std::optional<ConclusionOf<ast::BinaryExpression>> {
  switch (op) {
  case ast::BinaryExpression::BitAnd:
  case ast::BinaryExpression::BitOr:
  case ast::BinaryExpression::BitXor:
  case ast::BinaryExpression::ShiftLeft:
  case ast::BinaryExpression::ShiftRight:
    // Bit expressions always yield integer types.
    if (context.constraint == DynamaticTypingContext::FloatRequired)
      return std::nullopt;

    // Operands must be integer types.
    return ConclusionOf<ast::BinaryExpression>{
        {DynamaticTypingContext::IntegerRequired},
        {DynamaticTypingContext::IntegerRequired},
    };
  case ast::BinaryExpression::Greater:
  case ast::BinaryExpression::GreaterEqual:
  case ast::BinaryExpression::Less:
  case ast::BinaryExpression::LessEqual:
  case ast::BinaryExpression::Equal:
  case ast::BinaryExpression::NotEqual:
    // Equality operations always yield 'int'.
    if (context.constraint == DynamaticTypingContext::FloatRequired)
      return std::nullopt;
    [[fallthrough]];

  case ast::BinaryExpression::Plus:
  case ast::BinaryExpression::Minus:
  case ast::BinaryExpression::Mul:
    return Super::checkBinaryExpression(op, context);
  }
  llvm_unreachable("all enum values handled");
}

auto dynamatic::gen::DynamaticTypeSystem::checkUnaryExpression(
    ast::UnaryExpression::Op op, DynamaticTypingContext context) const
    -> std::optional<ConclusionOf<ast::UnaryExpression>> {
  switch (op) {
  case ast::UnaryExpression::BitwiseNot:
    // Requires an integer, produces an integer.
    if (context.constraint != DynamaticTypingContext::IntegerRequired)
      return std::nullopt;

    return context;
  case ast::UnaryExpression::BoolNot:
    // Can only be generated if an integer is required.
    if (context.constraint != DynamaticTypingContext::IntegerRequired)
      return std::nullopt;

    // However, the operand itself may be of any type since boolean conversions
    // are supported.
    return DynamaticTypingContext{
        random.fromEnum<DynamaticTypingContext::Constraint>()};
  case ast::UnaryExpression::Minus:
    return {context};
  }
  llvm_unreachable("all enum values handled");
}
