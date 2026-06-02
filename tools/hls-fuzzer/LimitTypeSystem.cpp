#include "LimitTypeSystem.h"

dynamatic::ProbabilityTable<dynamatic::gen::AbstractTypeSystem::ExpressionKey>
dynamatic::gen::LimitTypeSystem::getExpressionProbabilityTable(
    const LimitTypingContext &context) {
  // Default probabilities for expressions.
  // Most expressions are 100 times more likely to be generated than a
  // constant.
  // Variables are only 10 times more likely after a minimum expression
  // depth.
  // Conditional expressions too.
  std::vector<std::pair<ExpressionKey, std::size_t>> probs{
      {ast::Variable::Tag{}, context.expressionDepth < 2 ? 1 : 10},
      {ast::ConditionalExpression::Tag{}, 10},
      {ast::CastExpression::Tag{}, 100},
      {ast::ArrayReadExpression::Tag{}, 100}};

  for (auto op : enumRange<ast::BinaryExpression::Op>())
    probs.emplace_back(op, 100);

  for (auto op : enumRange<ast::UnaryExpression::Op>())
    probs.emplace_back(op, 100);

  return ProbabilityTable<ExpressionKey>(std::move(probs));
}
