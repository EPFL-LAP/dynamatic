#include "BasicCGenerator.h"

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
  if (inputType == to)
    return input;

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

auto gen::BasicCGenerator::generateFreshScalarParameter(
    ast::ScalarType datatype, const OpaqueContext &context)
    -> PendingParameter {
  scalarParameters.push_back(
      {{std::move(datatype), generateFreshVarName()}, context});
  return PendingParameter(*this, scalarParameters.back().first);
}

ast::ReturnStatement
gen::BasicCGenerator::generateReturnStatement(const OpaqueContext &context) {
  ast::Expression expression = generateExpression(context, 0);
  return ast::ReturnStatement{safeCastAsNeeded(
      llvm::cast<ast::ScalarType>(returnType), std::move(expression))};
}

constexpr std::size_t MAX_DEPTH = 4;

ast::Expression
gen::BasicCGenerator::generateExpression(const OpaqueContext &context,
                                         std::size_t depth) {
  using Constructor = std::function<std::optional<ast::Expression>(
      BasicCGenerator *, const OpaqueContext &, std::size_t)>;
  llvm::SmallVector<Constructor> generators;

  // Keep expressions interesting by making terminators less likely.
  if (depth > MAX_DEPTH || random.getSmallProbabilityBool())
    generators.emplace_back(&BasicCGenerator::generateConstant);
  if (depth > 2 || random.getRatherLowProbabilityBool())
    generators.emplace_back(&BasicCGenerator::generateScalarParameter);

  // Avoid stack overflows by restricting to a maximum expression depth.
  if (depth <= MAX_DEPTH) {
    for (auto op : enumRange<ast::BinaryExpression::Op>()) {
      generators.emplace_back([op](BasicCGenerator *self,
                                   const OpaqueContext &context,
                                   std::size_t depth) {
        return self->generateBinaryExpression(op, context, depth);
      });
    }
    generators.emplace_back(&BasicCGenerator::generateCastExpression);
    generators.emplace_back(&BasicCGenerator::generateArrayReadExpression);
    if (random.getRatherLowProbabilityBool())
      generators.emplace_back(&BasicCGenerator::generateConditionalExpression);
  }
  random.shuffle(generators);

  // If no other expression is allowed, then attempt to generate constants or
  // parameters rather than fail.
  // TODO: The entire logic here is a bit ad-hoc. We probably want probability
  //       tables that can be influenced by type systems somehow.
  generators.emplace_back(&BasicCGenerator::generateConstant);
  generators.emplace_back(&BasicCGenerator::generateScalarParameter);
  if (random.getBool())
    std::swap(generators.back(), generators[generators.size() - 2]);

  // Continuously generate an expression until one passes the type checker.
  for (Constructor &con : generators)
    if (std::optional<ast::Expression> result = con(this, context, depth))
      return std::move(*result);

  llvm_unreachable("it should always be possible to generate an expression");
}

std::optional<ast::Expression>
gen::BasicCGenerator::generateBinaryExpression(ast::BinaryExpression::Op op,
                                               const OpaqueContext &context,
                                               std::size_t depth) {
  auto conclusion = typeSystem.checkBinaryExpressionOpaque(op, context);
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

    std::optional<ast::ScalarType> scalarType = generateScalarType(
        context, /*toExclude=*/[&](const ast::ScalarType &value) {
          return !ast::BinaryExpression::isLegalOperandType(op, value);
        });
    if (!scalarType)
      return std::nullopt;

    lhs = safeCastAsNeeded(*scalarType, std::move(lhs));
    rhs = safeCastAsNeeded(*scalarType, std::move(rhs));
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
gen::BasicCGenerator::generateConditionalExpression(
    const OpaqueContext &context, std::size_t depth) {
  auto subcontext = typeSystem.checkConditionalExpressionOpaque(context);
  if (!subcontext)
    return std::nullopt;
  auto &&[cond, trueExpr, falseExpr] = *subcontext;

  return ast::ConditionalExpression{generateExpression(cond, depth + 1),
                                    generateExpression(trueExpr, depth + 1),
                                    generateExpression(falseExpr, depth + 1)};
}

std::optional<ast::CastExpression>
gen::BasicCGenerator::generateCastExpression(const OpaqueContext &context,
                                             std::size_t depth) {
  auto subcontext = typeSystem.checkCastExpressionOpaque(context);
  if (!subcontext)
    return std::nullopt;
  auto &&[typeCon, exprCon] = *subcontext;

  ast::Expression expression = generateExpression(exprCon, depth + 1);
  ast::ScalarType expressionType = expression.getType();

  // Keep it interesting by not performing noop-casts!
  std::optional<ast::ScalarType> datatype =
      generateScalarType(typeCon, /*toExclude=*/[&](auto &&value) {
        return value == expressionType;
      });
  if (!datatype)
    return std::nullopt;

  return ast::CastExpression{std::move(*datatype), std::move(expression)};
}

std::optional<ast::Constant>
gen::BasicCGenerator::generateConstant(const OpaqueContext &context,
                                       std::size_t) const {
  auto candidates = ast::PrimitiveType::ALL_PRIMITIVES;
  random.shuffle(candidates);

  for (ast::PrimitiveType::Type iter : candidates) {
    std::optional constant = [&] {
      switch (iter) {
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
    if (constant = typeSystem.checkConstantOpaque(*constant, context); constant)
      return constant;
  }
  return std::nullopt;
}

std::optional<ast::ArrayReadExpression>
gen::BasicCGenerator::generateArrayReadExpression(const OpaqueContext &context,
                                                  std::size_t depth) {
  auto conclusion = typeSystem.checkArrayReadExpressionOpaque(context);
  if (!conclusion)
    return std::nullopt;
  auto [paramConc, indexConc] = *conclusion;

  // Construct a safe indexing expression from an array parameter.
  auto genWrappedArrayReadFromParam = [&, &indexConc = indexConc](
                                          const ast::ArrayParameter &param) {
    ast::ScalarType elementType = param.getElementType();
    std::size_t mask = param.getDimension() - 1;
    std::string name = param.getName().str();
    // Generate an indexing expression.
    // Has to be an integer.
    ast::Expression index = safeCastAsNeeded(
        ast::PrimitiveType::UInt32, generateExpression(indexConc, depth + 1));

    // Bitmask the index to be in range of the array! We use this to avoid
    // undefined behavior in our programs. In the future we could also add
    // mechanisms (type systems, or whatever), that restrict expressions to
    // safe in-range expressions.
    //
    // Note: We can use a bitmask here since array parameters that we generate
    // are all powers-of-2. We do so since the modulo operator is currently
    // unsupported in dynamatic.
    return ast::ArrayReadExpression{
        std::move(elementType), name,
        ast::BinaryExpression{std::move(index), ast::BinaryExpression::BitAnd,
                              ast::Constant{static_cast<std::uint32_t>(mask)}}};
  };

  std::optional<ast::ArrayParameter> arrayParameter =
      generateArrayParameter(paramConc);
  if (!arrayParameter)
    return std::nullopt;
  return genWrappedArrayReadFromParam(*arrayParameter);
}

std::optional<ast::ArrayParameter>
gen::BasicCGenerator::generateArrayParameter(const OpaqueContext &context,
                                             std::size_t depth) {
  // With a low chance, skip picking an existing parameter and try to generate
  // a new one.
  if (!random.getRatherLowProbabilityBool()) {
    // Randomly shuffle the parameter ordering and find the first parameter
    // that passes type checking.
    std::vector<ast::ArrayParameter> copy(arrayParameters.size());
    llvm::copy(llvm::make_first_range(arrayParameters), copy.begin());
    random.shuffle(copy);

    for (const ast::ArrayParameter &candidateParam : copy)
      if (typeSystem.checkArrayParameterOpaque(candidateParam, context))
        return candidateParam;
  }

  std::optional<ast::ScalarType> elementType = generateScalarType(context);
  if (!elementType)
    return std::nullopt;

  arrayParameters.push_back(
      {{std::move(*elementType), generateFreshVarName(),
        // Generate a power-of-2 dimension to make the modulo operator fast and
        // easy to implement.
        // We choose an arbitrary upper-bound of 32 for the dimension for now.
        static_cast<std::size_t>(1 << random.getInteger(0, 5))},
       context});
  if (!typeSystem.checkArrayParameterOpaque(arrayParameters.back().first,
                                            context)) {
    arrayParameters.pop_back();
    varCounter--;
    return std::nullopt;
  }
  return arrayParameters.back().first;
}

std::optional<ast::Variable>
gen::BasicCGenerator::generateScalarParameter(const OpaqueContext &context,
                                              std::size_t) {
  auto conclusion = typeSystem.checkVariableOpaque(context);
  if (!conclusion)
    return std::nullopt;

  // With a low chance, skip picking an existing parameter and try to generate
  // a new one.
  if (!random.getRatherLowProbabilityBool()) {
    // Randomly shuffle the parameter ordering and find the first parameter
    // that passes type checking.
    std::vector<ast::ScalarParameter> copy(scalarParameters.size());
    llvm::copy(llvm::make_first_range(scalarParameters), copy.begin());
    random.shuffle(copy);

    for (ast::ScalarParameter &iter : copy)
      if (typeSystem.checkScalarParameterOpaque(iter, *conclusion))
        return ast::Variable{iter.getDataType(), iter.getName().str()};
  }

  std::optional<ast::ScalarType> datatype = generateScalarType(*conclusion);
  if (!datatype)
    return std::nullopt;

  PendingParameter pendingParam =
      generateFreshScalarParameter(*datatype, context);
  if (typeSystem.checkScalarParameterOpaque(pendingParam.getParameter(),
                                            *conclusion)) {
    ast::ScalarParameter parameter = pendingParam.commit();
    return ast::Variable{parameter.getDataType(), parameter.getName().str()};
  }
  return std::nullopt;
}

std::optional<ast::ScalarType> gen::BasicCGenerator::generateScalarType(
    const OpaqueContext &context,
    llvm::function_ref<bool(const ast::ScalarType &)> toExclude) const {
  auto candidates = ast::PrimitiveType::ALL_PRIMITIVES;
  random.shuffle(candidates);
  for (ast::ScalarType iter : candidates) {
    // Skip some types based on the caller excluding them.
    if (toExclude && toExclude(iter))
      continue;

    if (typeSystem.checkScalarTypeOpaque(iter, context))
      return iter;
  }

  return std::nullopt;
}

ast::ReturnType
gen::BasicCGenerator::generateReturnType(const OpaqueContext &context) const {
  // Candidates for return types are all primitive types as well as 'void'.
  // (i.e., one more than the number of primitive types).
  std::array<ast::ReturnType, ast::PrimitiveType::ALL_PRIMITIVES.size() + 1>
      candidates;
  llvm::copy(ast::PrimitiveType::ALL_PRIMITIVES, candidates.begin());
  candidates.back() = ast::VoidType{};
  random.shuffle(candidates);
  for (const ast::ReturnType &iter : candidates)
    if (typeSystem.checkReturnTypeOpaque(iter, context))
      return iter;

  llvm::report_fatal_error(
      "It must always be possible to generate a return type");
}

constexpr std::size_t MAX_STATEMENTS = 10;

std::vector<ast::Statement>
gen::BasicCGenerator::generateStatementList(const OpaqueContext &context) {
  std::vector<ast::Statement> result;
  // TODO: Type systems should have better control over the number of
  //       statements and in what order they are generated.
  //       Right now they are always generated last statement to first.
  std::size_t numStatements = random.getInteger<std::size_t>(0, MAX_STATEMENTS);
  result.reserve(numStatements);
  for (std::size_t i = 0; i < numStatements; i++) {
    std::optional<ast::Statement> maybeStat = generateStatement(context);
    if (!maybeStat)
      break;

    result.push_back(std::move(*maybeStat));
  }
  std::reverse(result.begin(), result.end());
  return result;
}

std::optional<ast::Statement>
gen::BasicCGenerator::generateStatement(const OpaqueContext &context) {
  return generateArrayAssignmentStatement(context);
}

std::optional<ast::ArrayAssignmentStatement>
gen::BasicCGenerator::generateArrayAssignmentStatement(
    const OpaqueContext &context) {
  auto conclusion = typeSystem.checkArrayAssignmentStatementOpaque(context);
  if (!conclusion)
    return std::nullopt;

  auto &&[param, index, value] = *conclusion;
  std::optional<ast::ArrayParameter> parameter = generateArrayParameter(param);
  if (!parameter)
    return std::nullopt;

  ast::Expression castAsNeeded = safeCastAsNeeded(
      /*to=*/ast::PrimitiveType::UInt32,
      generateExpression(/*context=*/index, /*depth=*/0));
  castAsNeeded = ast::BinaryExpression{
      std::move(castAsNeeded), ast::BinaryExpression::BitAnd,
      ast::Constant{static_cast<std::uint32_t>(parameter->getDimension() - 1)}};
  return ast::ArrayAssignmentStatement{
      parameter->getName().str(),
      castAsNeeded,
      generateExpression(value, 0),
  };
}

ast::Function gen::BasicCGenerator::generate(std::string_view functionName) {
  auto conclusion = typeSystem.checkFunctionOpaque(entryContext);
  returnType = generateReturnType(conclusion.returnType);
  std::optional<ast::ReturnStatement> returnStatement;
  if (!std::holds_alternative<ast::VoidType>(returnType))
    returnStatement = generateReturnStatement(conclusion.returnStatement);

  std::vector<ast::Statement> statementList =
      generateStatementList(conclusion.returnStatement);

  auto scalarRange = llvm::make_first_range(scalarParameters);
  auto arrayRange = llvm::make_first_range(arrayParameters);
  return ast::Function{
      returnType,
      std::string(functionName),
      std::vector(scalarRange.begin(), scalarRange.end()),
      std::vector(arrayRange.begin(), arrayRange.end()),
      statementList,
      std::move(returnStatement),
  };
}

std::string
gen::BasicCGenerator::generateTestBench(const ast::Function &kernel) const {
  std::string s;
  llvm::raw_string_ostream ss(s);
  ss << "\nint main() {\n";
  mlir::raw_indented_ostream os(ss);
  os.indent();
  for (const auto &[parameter, context] : scalarParameters) {
    std::optional<ast::Constant> constant;
    while (!constant) {
      constant = generateConstant(context);
    }

    os << parameter.getDataType() << ' ' << parameter.getName() << " = "
       << *constant << ";\n";
  }

  for (const auto &[parameter, context] : arrayParameters) {
    os << parameter.getElementType() << ' ' << parameter.getName() << "["
       << parameter.getDimension() << "] = {";
    llvm::interleaveComma(
        llvm::seq<std::size_t>(0, parameter.getDimension()), os,
        [&, &context = context, &parameter = parameter](auto &&) {
          std::optional<ast::Constant> constant;
          while (!constant) {
            constant = generateConstant(context);
          }
          // C++ does not allow implicit casts in array constructors, so we
          // must cast the constant explicitly.
          os << safeCastAsNeeded(parameter.getElementType(), *constant);
        });
    os << "};\n";
  }

  os << "CALL_KERNEL(" << kernel.name;
  for (const ast::ScalarParameter &iter : kernel.scalarParameters) {
    os << ", " << iter.getName();
  }
  for (const ast::ArrayParameter &iter : kernel.arrayParameters) {
    os << ", " << iter.getName();
  }
  os << ");";
  ss << "\n}\n";
  return s;
}
