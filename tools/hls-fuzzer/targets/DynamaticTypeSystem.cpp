#include "DynamaticTypeSystem.h"

bool dynamatic::gen::DynamaticTypeSystem::discardScalarType(
    const ast::ScalarType &scalarType, DynamaticTypingContext context) {
  switch (context.constraint) {
  case DynamaticTypingContext::FloatRequired:
    return scalarType != ast::PrimitiveType::Float &&
           scalarType != ast::PrimitiveType::Double;

  case DynamaticTypingContext::IntegerRequired:
    return scalarType == ast::PrimitiveType::Float ||
           scalarType == ast::PrimitiveType::Double;
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
dynamatic::gen::DynamaticTypeSystem::getBinaryExpressionTransferFns(
    ast::BinaryExpression::Op op) {
  switch (op) {
  case ast::BinaryExpression::BitAnd:
  case ast::BinaryExpression::BitOr:
  case ast::BinaryExpression::BitXor:
  case ast::BinaryExpression::ShiftLeft:
  case ast::BinaryExpression::ShiftRight:
    return {/*lhs=*/TransferFn<ast::BinaryExpression>(DynamaticTypingContext{
                DynamaticTypingContext::IntegerRequired}),
            /*rhs=*/
            TransferFn<ast::BinaryExpression>(DynamaticTypingContext{
                DynamaticTypingContext::IntegerRequired}),
            /*output=*/
            TransferFn<ast::BinaryExpression>(DynamaticTypingContext{
                DynamaticTypingContext::IntegerRequired})};
  default:
    return Super::getBinaryExpressionTransferFns(op);
  }
}

dynamatic::gen::TransferFnArray<dynamatic::ast::UnaryExpression>
dynamatic::gen::DynamaticTypeSystem::getUnaryExpressionTransferFns(
    ast::UnaryExpression::Op op) {
  switch (op) {
  case ast::UnaryExpression::BitwiseNot:
    return {
        /*operand=*/TransferFn<ast::UnaryExpression>(
            DynamaticTypingContext{DynamaticTypingContext::IntegerRequired}),
        /*output=*/copyFromParent<ast::UnaryExpression>(),
    };
  case ast::UnaryExpression::BoolNot:
    return {
        /*operand=*/TransferFn<ast::UnaryExpression>(DynamaticTypingContext{
            random.fromEnum<DynamaticTypingContext::Constraint>()}),
        /*output=*/copyFromParent<ast::UnaryExpression>(),
    };
  case ast::UnaryExpression::Minus:
    return {
        /*operand=*/copyFromParent<ast::UnaryExpression>(),
        /*output=*/copyFromParent<ast::UnaryExpression>(),
    };
  }
  llvm_unreachable("all enum cases handled");
}

bool dynamatic::gen::DynamaticTypeSystem::discardUnaryExpression(
    ast::UnaryExpression::Op op, DynamaticTypingContext context) {
  switch (op) {
  case ast::UnaryExpression::BitwiseNot:
  case ast::UnaryExpression::BoolNot:
    // Can't apply to float.
    return context.constraint == DynamaticTypingContext::FloatRequired;
  case ast::UnaryExpression::Minus:
    return false;
  }
  llvm_unreachable("all enum values handled");
}
