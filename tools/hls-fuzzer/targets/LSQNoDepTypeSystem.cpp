#include "LSQNoDepTypeSystem.h"

std::optional<dynamatic::ast::ArrayReadExpression>
dynamatic::gen::LSQNoDepTypeSystem::generateArrayReadExpression(
    const Context &context, GenerateCallback<ast::ArrayParameter, Context>,
    GenerateCallback<ast::Expression, Context>) {
  auto &lsqNoDepContext = std::get<LSQNoDepContext>(context);
  if (!lsqNoDepContext.elementWritten)
    return std::nullopt;

  // Make sure the type system allows us to even read from this array parameter.
  if (!checkExistingArrayParameter(*lsqNoDepContext.elementWritten->first,
                                   context))
    return std::nullopt;

  // We only generate array reads from the array being written to at the
  // moment.
  return ast::ArrayReadExpression{
      lsqNoDepContext.elementWritten->first->getElementType(),
      lsqNoDepContext.elementWritten->first->getName().str(),
      lsqNoDepContext.elementWritten->second};
}

std::optional<dynamatic::ast::ArrayAssignmentStatement>
dynamatic::gen::LSQNoDepTypeSystem::generateArrayAssignmentStatement(
    const Context &context,
    GenerateCallback<ast::ArrayParameter, Context> generateArrayParameter,
    GenerateCallback<ast::Expression, Context> generateExpression) {
  auto conclusion = checkArrayAssignmentStatement(context);
  if (!conclusion)
    return std::nullopt;
  auto &&[param, index, value] = std::move(*conclusion);

  std::optional<ast::ArrayParameter> parameter = generateArrayParameter(param);
  if (!parameter)
    return std::nullopt;

  assert(llvm::isPowerOf2_64(parameter->getDimension()) &&
         "default implementation depends on dimensions being powers of 2");

  std::optional<ast::Expression> maybeIndexingExpression =
      generateExpression(index);
  if (!maybeIndexingExpression)
    return std::nullopt;

  ast::BinaryExpression indexExpression{
      std::move(*maybeIndexingExpression), ast::BinaryExpression::BitAnd,
      ast::Constant{static_cast<std::uint32_t>(parameter->getDimension() - 1)}};

  // Potentially generate a data-dependent RAW dependency by requiring
  // array-read expressions to read from the array we're writing to.
  auto &lsqNoDepContext = std::get<LSQNoDepContext>(value);
  lsqNoDepContext.elementWritten.emplace(&*parameter, indexExpression);
  std::optional<ast::Expression> valueExpression = generateExpression(value);
  if (!valueExpression)
    return std::nullopt;

  return ast::ArrayAssignmentStatement{
      parameter->getName().str(),
      indexExpression,
      std::move(*valueExpression),
  };
}
