#include "BasicCProducer.h"

#include "mlir/Support/IndentedOstream.h"

#include <functional>
#include <sstream>

using namespace dynamatic;

static ast::Expression generateMinExpression(const ast::Expression &lhs,
                                             const ast::Expression &rhs) {
  return ast::ConditionalExpression{
      ast::BinaryExpression{lhs, ast::BinaryExpression::Less, rhs}, lhs, rhs};
}

static ast::Expression generateMaxExpression(const ast::Expression &lhs,
                                             const ast::Expression &rhs) {
  return ast::ConditionalExpression{
      ast::BinaryExpression{lhs, ast::BinaryExpression::Greater, rhs}, lhs,
      rhs};
}

/// Performs a cast from whatever type 'input' to 'to' in a manner that does not
/// trigger undefined behavior.
static ast::Expression safeCastAsNeeded(const ast::ScalarType &to,
                                        ast::Expression input) {
  ast::ScalarType inputType = input.getType();
  // Only casts that can cause undefined behavior are casts from floating point
  // types to integers.
  auto *inputPrim = llvm::dyn_cast<ast::PrimitiveType>(inputType);
  if (!inputPrim || inputPrim->isInteger())
    return ast::CastExpression{to, std::move(input)};

  // Input known to be a floating point type now.
  auto *outputPrim = llvm::dyn_cast<ast::PrimitiveType>(inputType);
  if (outputPrim && !outputPrim->isInteger())
    // Can safely cast from float to float.
    return ast::CastExpression{to, std::move(input)};

  assert(outputPrim && "only scalar type are currently primitives");

  // Float to integer casts require clamping the value to be in range of the
  // integer.
  return generateMaxExpression(
      generateMinExpression(input, outputPrim->getMaxValue()),
      outputPrim->getMinValue());
}

const ast::Parameter &
gen::BasicCProducer::generateFreshParameter(ast::ScalarType datatype,
                                            const OpaqueContext &context) {
  parameters.push_back(
      {{std::move(datatype), generateFreshVarName()}, context});
  return parameters.back().first;
}

ast::ReturnStatement
gen::BasicCProducer::generateFunctionBody(const OpaqueContext &constraints) {
  ast::Expression expression = generateExpression(constraints, 0);
  return ast::ReturnStatement{
      safeCastAsNeeded(returnType, std::move(expression))};
}

constexpr std::size_t MAX_DEPTH = 8;

ast::Expression
gen::BasicCProducer::generateExpression(const OpaqueContext &constraint,
                                        std::size_t depth) {
  using Constructor = std::function<std::optional<ast::Expression>(
      BasicCProducer *, const OpaqueContext &, std::size_t)>;
  std::vector<Constructor> generators;

  // Continuously generate an expression until one passes the type checker.
  while (true) {
    generators.clear();

    // Keep expressions interesting by making terminators less likely.
    if (depth > MAX_DEPTH || random.getSmallProbabilityBool())
      generators.emplace_back(&BasicCProducer::generateConstant);
    if (depth > 2)
      generators.emplace_back(&BasicCProducer::generateScalarParameter);

    // Avoid stack overflows by restricting to a maximum expression depth.
    if (depth <= MAX_DEPTH) {
      generators.emplace_back(&BasicCProducer::generateBinaryExpression);
      generators.emplace_back(&BasicCProducer::generateCastExpression);
      if (random.getRatherLowProbabilityBool())
        generators.emplace_back(&BasicCProducer::generateConditionalExpression);
    }

    std::vector<Constructor> subset = random.getNonEmptySubset(generators);
    for (const Constructor &iter : subset)
      if (std::optional<ast::Expression> result = iter(this, constraint, depth))
        return std::move(*result);
  }
}

std::optional<ast::Expression>
gen::BasicCProducer::generateBinaryExpression(const OpaqueContext &constraints,
                                              std::size_t depth) {

  auto op = random.fromEnum<ast::BinaryExpression::Op>();
  auto conclusion = typeSystem.checkBinaryExpressionOpaque(op, constraints);
  if (!conclusion)
    return std::nullopt;
  auto [lhsCons, rhsCons] = *conclusion;

  ast::Expression lhs = generateExpression(lhsCons, depth + 1);
  ast::Expression rhs = generateExpression(rhsCons, depth + 1);

  // Perform explicit casts to a legal operand type if neither of the
  // expressions are legal for the given operation.
  // This would e.g. cast 'double's that are meant to be applied to '&' to a
  // random type that can be legally used with '&'.
  if (!ast::BinaryExpression::isLegalOperandType(op, lhs.getType()) ||
      !ast::BinaryExpression::isLegalOperandType(op, rhs.getType())) {
    ast::ScalarType scalarType;
    do {
      scalarType = generateScalarType(constraints);
    } while (!ast::BinaryExpression::isLegalOperandType(op, scalarType));
    lhs = safeCastAsNeeded(scalarType, std::move(lhs));
    rhs = safeCastAsNeeded(scalarType, std::move(rhs));
  }

  switch (op) {
  case ast::BinaryExpression::ShiftLeft:
  case ast::BinaryExpression::ShiftRight: {
    ast::ScalarType datatype = lhs.getType();
    // Restrict the right expression to be in range of the bitwidth.
    rhs = ast::BinaryExpression{
        std::move(rhs), ast::BinaryExpression::BitAnd,
        ast::Constant{static_cast<uint32_t>(datatype.getBitwidth() - 1)}};

    // If the left-hand side is a signed integer, make sure the value is at
    // least 0.
    // Performing a left-shift on a negative value in C is undefined behavior.
    if (op == ast::BinaryExpression::ShiftLeft && datatype.isSigned())
      lhs = generateMinExpression(std::move(lhs),
                                  ast::Constant{static_cast<uint32_t>(0)});
    return ast::BinaryExpression{std::move(lhs), op, std::move(rhs)};
  }
  case ast::BinaryExpression::Plus:
  case ast::BinaryExpression::Minus:
  case ast::BinaryExpression::Mul: {
    ast::ScalarType lhsType = lhs.getType();
    ast::ScalarType rhsType = rhs.getType();
    if ((lhsType == ast::PrimitiveType::Int32 &&
         lhsType.getBitwidth() > rhsType.getBitwidth()) ||
        (rhsType == ast::PrimitiveType::Int32 &&
         rhsType.getBitwidth() > lhsType.getBitwidth())) {
      // Promote integers where one operand is an 'int32_t' to 'uint32_t' to
      // avoid undefined behavior on overflow.
      lhs = safeCastAsNeeded(ast::PrimitiveType::UInt32, std::move(lhs));
      rhs = safeCastAsNeeded(ast::PrimitiveType::UInt32, std::move(rhs));
    }
    return ast::BinaryExpression{std::move(lhs), op, std::move(rhs)};
  }
  // case ast::BinaryExpression::Division:
  break;
  case ast::BinaryExpression::BitAnd:
  case ast::BinaryExpression::BitOr:
  case ast::BinaryExpression::BitXor:
  case ast::BinaryExpression::Greater:
  case ast::BinaryExpression::GreaterEqual:
  case ast::BinaryExpression::Less:
  case ast::BinaryExpression::LessEqual:
  case ast::BinaryExpression::Equal:
  case ast::BinaryExpression::NotEqual:
    return ast::BinaryExpression{std::move(lhs), op, std::move(rhs)};
  }
  llvm_unreachable("all enum cases handled");
}

std::optional<ast::ConditionalExpression>
gen::BasicCProducer::generateConditionalExpression(
    const OpaqueContext &constraint, std::size_t depth) {
  auto subConstraints = typeSystem.checkConditionalExpressionOpaque(constraint);
  if (!subConstraints)
    return std::nullopt;
  auto &&[cond, trueExpr, falseExpr] = *subConstraints;

  return ast::ConditionalExpression{generateExpression(cond, depth + 1),
                                    generateExpression(trueExpr, depth + 1),
                                    generateExpression(falseExpr, depth + 1)};
}

std::optional<ast::CastExpression>
gen::BasicCProducer::generateCastExpression(const OpaqueContext &constraint,
                                            std::size_t depth) {
  auto subConstraints = typeSystem.checkCastExpressionOpaque(constraint);
  if (!subConstraints)
    return std::nullopt;
  auto &&[typeCon, exprCon] = *subConstraints;

  ast::Expression expression = generateExpression(exprCon, depth + 1);
  ast::ScalarType expressionType = expression.getType();

  // Keep it interesting by not performing noop-casts!
  ast::ScalarType datatype = generateScalarType(typeCon);
  while (datatype == expressionType)
    datatype = generateScalarType(typeCon);

  return ast::CastExpression{std::move(datatype), std::move(expression)};
}

std::optional<ast::Constant>
gen::BasicCProducer::generateConstant(const OpaqueContext &constraints,
                                      std::size_t) const {
  ast::Constant constant = [&] {
    switch (random.fromEnum<ast::PrimitiveType::Type>()) {
    case ast::PrimitiveType::Int8:
      return ast::Constant{random.getInterestingInteger<std::int8_t>()};
    case ast::PrimitiveType::UInt8:
      return ast::Constant{random.getInterestingInteger<std::uint8_t>()};

    case ast::PrimitiveType::Int16:
      return ast::Constant{random.getInterestingInteger<std::int16_t>()};

    case ast::PrimitiveType::UInt16:
      return ast::Constant{random.getInterestingInteger<std::uint16_t>()};

    case ast::PrimitiveType::Int32:
      return ast::Constant{random.getInterestingInteger<std::int32_t>()};

    case ast::PrimitiveType::UInt32:
      return ast::Constant{random.getInterestingInteger<std::uint32_t>()};

    case ast::PrimitiveType::Float:
      return ast::Constant{random.getInterestingFloat()};
    case ast::PrimitiveType::Double:
      return ast::Constant{random.getInterestingDouble()};
    }
    llvm_unreachable("all enum cases handled");
  }();
  if (typeSystem.checkConstantOpaque(constant, constraints))
    return constant;

  return std::nullopt;
}

std::optional<ast::Variable>
gen::BasicCProducer::generateScalarParameter(const OpaqueContext &constraints,
                                             std::size_t) {
  auto conclusion = typeSystem.checkVariableOpaque(constraints);
  if (!conclusion)
    return std::nullopt;

  ast::Parameter parameter = [&] {
    if (parameters.empty() || random.getRatherLowProbabilityBool())
      return generateFreshParameter(generateScalarType(*conclusion),
                                    constraints);

    // Attempt to find a random parameter that makes the type system happy.
    // The current number of parameter is used as an arbitrary heuristic as to
    // how many attempts we should perform.
    for (std::size_t i = 0; i < parameters.size(); i++) {
      ast::Parameter &choice = random.fromRange(parameters).first;
      if (typeSystem.checkParameterOpaque(choice, *conclusion))
        return choice;
    }

    // Otherwise we fall back to a random parameter.
    return generateFreshParameter(generateScalarType(*conclusion), constraints);
  }();

  return ast::Variable{parameter.datatype, parameter.name};
}

ast::ScalarType gen::BasicCProducer::generateScalarType(
    const OpaqueContext &constraints) const {
  while (true) {
    ast::ScalarType datatype = random.fromEnum<ast::PrimitiveType::Type>();
    if (typeSystem.checkScalarTypeOpaque(datatype, constraints))
      return datatype;
  }
}

ast::Function gen::BasicCProducer::generate(std::string_view functionName) {
  auto conclusion = typeSystem.checkFunctionOpaque(entryContext);

  returnType = generateScalarType(conclusion.returnType);
  ast::ReturnStatement body = generateFunctionBody(conclusion.returnStatement);
  auto range = llvm::make_first_range(parameters);
  return ast::Function{
      returnType,
      std::string(functionName),
      std::vector(range.begin(), range.end()),
      std::move(body),
  };
}

std::string
gen::BasicCProducer::generateTestBench(const ast::Function &kernel) const {
  std::string s;
  llvm::raw_string_ostream ss(s);
  ss << "\nint main() {\n";
  mlir::raw_indented_ostream os(ss);
  os.indent();
  for (const auto &[parameter, constraint] : parameters) {
    std::optional<ast::Constant> constant;
    while (!constant) {
      constant = generateConstant(constraint);
    }

    auto &[datatype, name] = parameter;
    os << ast::PrintTypePrefix{datatype} << ' ' << name
       << ast::PrintTypeSuffix{datatype} << " = " << *constant << ";\n";
  }
  os << "CALL_KERNEL(" << kernel.name;
  for (const auto &[datatype, name] : kernel.parameters) {
    os << ", " << name;
  }
  os << ");";
  ss << "\n}\n";
  return s;
}
