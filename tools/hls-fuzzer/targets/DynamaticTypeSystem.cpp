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

bool dynamatic::gen::DynamaticTypeSystem::discardBinaryExpression(
    ast::BinaryExpression::Op op, DynamaticTypingContext context) {
  switch (op) {
  case ast::BinaryExpression::BitAnd:
  case ast::BinaryExpression::BitOr:
  case ast::BinaryExpression::BitXor:
  case ast::BinaryExpression::ShiftLeft:
  case ast::BinaryExpression::ShiftRight:
    // Bit expressions always yield integer types.
    return context.constraint == DynamaticTypingContext::FloatRequired;

  case ast::BinaryExpression::Greater:
  case ast::BinaryExpression::GreaterEqual:
  case ast::BinaryExpression::Less:
  case ast::BinaryExpression::LessEqual:
  case ast::BinaryExpression::Equal:
  case ast::BinaryExpression::NotEqual:
    // Equality operations always yield 'int'.
    return context.constraint == DynamaticTypingContext::FloatRequired;
  case ast::BinaryExpression::Plus:
  case ast::BinaryExpression::Minus:
  case ast::BinaryExpression::Mul:
    return false;
  }
  llvm_unreachable("all enum values handled");
}

dynamatic::gen::TransferFnArray<dynamatic::ast::BinaryExpression>
dynamatic::gen::DynamaticTypeSystem::getBinaryExpressionContextDependencies(
    ast::BinaryExpression::Op op) {
  switch (op) {
  case ast::BinaryExpression::BitAnd:
  case ast::BinaryExpression::BitOr:
  case ast::BinaryExpression::BitXor:
  case ast::BinaryExpression::ShiftLeft:
  case ast::BinaryExpression::ShiftRight:
    return {/*lhs=*/Dependency<ast::BinaryExpression>(DynamaticTypingContext{
                DynamaticTypingContext::IntegerRequired}),
            /*rhs=*/
            Dependency<ast::BinaryExpression>(DynamaticTypingContext{
                DynamaticTypingContext::IntegerRequired}),
            /*output=*/
            Dependency<ast::BinaryExpression>(DynamaticTypingContext{
                DynamaticTypingContext::IntegerRequired})};
  default:
    return Super::getBinaryExpressionContextDependencies(op);
  }
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
