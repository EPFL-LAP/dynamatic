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

std::pair<ast::ReturnStatement, gen::OpaqueContext>
gen::BasicCGenerator::generateReturnStatement(const OpaqueContext &context) {
  return *generateWithDependencies<ast::ReturnStatement>(
      context, typeSystem.getReturnStatementTransferFns(),
      /*return value=*/
      [&](const OpaqueContext &context) {
        auto [expression, outputContext] = generateExpression(context, 0);
        if (maybeReturnType && llvm::isa<ast::ScalarType>(*maybeReturnType))
          expression =
              safeCastAsNeeded(llvm::cast<ast::ScalarType>(*maybeReturnType),
                               std::move(expression));
        return std::pair{std::move(expression), std::move(outputContext)};
      },
      /*constructor=*/
      [&](ast::Expression &&expression) {
        return ast::ReturnStatement{std::move(expression)};
      });
}

constexpr std::size_t MAX_DEPTH = 4;

std::pair<ast::Expression, gen::OpaqueContext>
gen::BasicCGenerator::generateExpression(const OpaqueContext &context,
                                         std::size_t depth) {
  using Constructor =
      std::function<std::optional<std::pair<ast::Expression, OpaqueContext>>(
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
    for (auto op : enumRange<ast::UnaryExpression::Op>()) {
      generators.emplace_back([op](BasicCGenerator *self,
                                   const OpaqueContext &context,
                                   std::size_t depth) {
        return self->generateUnaryExpression(op, context, depth);
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
    if (std::optional<std::pair<ast::Expression, OpaqueContext>> result =
            con(this, context, depth))
      return std::move(*result);

  llvm_unreachable("it should always be possible to generate an expression");
}

std::optional<std::pair<ast::Expression, gen::OpaqueContext>>
gen::BasicCGenerator::generateBinaryExpression(ast::BinaryExpression::Op op,
                                               const OpaqueContext &context,
                                               std::size_t depth) {
  if (typeSystem.discardBinaryExpressionOpaque(op, context))
    return std::nullopt;

  return generateWithDependencies<ast::BinaryExpression>(
      context, typeSystem.getBinaryExpressionTransferFns(op),
      /*lhs=*/
      [&](const OpaqueContext &context) {
        return generateExpression(context, depth + 1);
      },
      /*rhs=*/
      [&](const OpaqueContext &context) {
        return generateExpression(context, depth + 1);
      },
      /*constructor=*/
      [&](ast::Expression &&lhs,
          ast::Expression &&rhs) -> std::optional<ast::BinaryExpression> {
        switch (op) {
        case ast::BinaryExpression::ShiftLeft:
        case ast::BinaryExpression::ShiftRight: {
          ast::ScalarType datatype = lhs.getType();
          // Restrict the right expression to be in range of the bitwidth.
          rhs = ast::BinaryExpression{
              std::move(rhs), ast::BinaryExpression::BitAnd,
              ast::Constant{static_cast<uint32_t>(datatype.getBitwidth() - 1)}};

          // If the left-hand side is a signed integer, make sure the value is
          // at least 0. Performing a left-shift on a negative value in C is
          // undefined behavior.
          if (op == ast::BinaryExpression::ShiftLeft && datatype.isSigned())
            lhs = generateMinExpression(
                std::move(lhs), ast::Constant{static_cast<uint32_t>(0)});
          return ast::BinaryExpression{std::move(lhs), op, std::move(rhs)};
        }
        case ast::BinaryExpression::Plus:
        case ast::BinaryExpression::Minus: {
          ast::ScalarType lhsType = lhs.getType();
          ast::ScalarType rhsType = rhs.getType();
          if (lhsType.isInteger() && rhsType.isInteger() &&
              std::max(lhsType.getBitwidth(), rhsType.getBitwidth()) + 1 >=
                  32) {
            // Explicitly promote integers to 'uint32_t' if the operation may
            // overflow to avoid undefined behavior.
            // Otherwise, the operation is performed on 'int32_t' due to C's
            // promotion rules, which has undefined behavior on overflow.
            //
            // Note that in LLVM IR signed and unsigned multiplications are
            // identical operations except for the wraparound behaviour for
            // unsigned. Signed overflow is defined to be poison via the 'nsw'
            // flag.
            lhs = safeCastAsNeeded(ast::PrimitiveType::UInt32, std::move(lhs));
            rhs = safeCastAsNeeded(ast::PrimitiveType::UInt32, std::move(rhs));
          }
          return ast::BinaryExpression{std::move(lhs), op, std::move(rhs)};
        }
        case ast::BinaryExpression::Mul: {
          ast::ScalarType lhsType = lhs.getType();
          ast::ScalarType rhsType = rhs.getType();
          if (lhsType.isInteger() && rhsType.isInteger() &&
              lhsType.getBitwidth() + rhsType.getBitwidth() >= 32) {
            // Explicitly promote integers to 'uint32_t' if the operation may
            // overflow to avoid undefined behavior.
            // Otherwise, the operation is performed on 'int32_t' due to C's
            // promotion rules, which has undefined behavior on overflow.
            //
            // Note that in LLVM IR signed and unsigned multiplications are
            // identical operations except for the wraparound behaviour for
            // unsigned. Signed overflow is defined to be poison via the 'nsw'
            // flag.
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
      });
}

std::optional<std::pair<ast::Expression, gen::OpaqueContext>>
gen::BasicCGenerator::generateUnaryExpression(ast::UnaryExpression::Op op,
                                              const OpaqueContext &context,
                                              std::size_t depth) {
  if (typeSystem.discardUnaryExpressionOpaque(op, context))
    return std::nullopt;

  return generateWithDependencies<ast::UnaryExpression>(
      context, typeSystem.getUnaryExpressionTransferFns(op),
      /*operand=*/
      [&](const OpaqueContext &context) {
        return generateExpression(context, depth + 1);
      },
      /*constructor=*/
      [&](ast::Expression &&operand) -> std::optional<ast::UnaryExpression> {
        return ast::UnaryExpression{op, std::move(operand)};
      });
}

std::optional<std::pair<ast::ConditionalExpression, gen::OpaqueContext>>
gen::BasicCGenerator::generateConditionalExpression(
    const OpaqueContext &context, std::size_t depth) {
  if (typeSystem.discardConditionalExpressionOpaque(context))
    return std::nullopt;

  return generateWithDependencies<ast::ConditionalExpression>(
      context, typeSystem.getConditionalExpressionTransferFns(),
      /*condition=*/
      [&](const OpaqueContext &context) {
        return generateExpression(context, depth + 1);
      },
      /*true value=*/
      [&](const OpaqueContext &context) {
        return generateExpression(context, depth + 1);
      },
      /*false value=*/
      [&](const OpaqueContext &context) {
        return generateExpression(context, depth + 1);
      },
      /*constructor=*/
      [&](ast::Expression &&cond, ast::Expression &&trueExpr,
          ast::Expression &&falseExpr) {
        return ast::ConditionalExpression{std::move(cond), std::move(trueExpr),
                                          std::move(falseExpr)};
      });
}

std::optional<std::pair<ast::CastExpression, gen::OpaqueContext>>
gen::BasicCGenerator::generateCastExpression(const OpaqueContext &context,
                                             std::size_t depth) {
  if (typeSystem.discardCastExpressionOpaque(context))
    return std::nullopt;

  return generateWithDependencies<ast::CastExpression>(
      context, typeSystem.getCastExpressionTransferFns(),
      /*data type=*/
      [&](const OpaqueContext &context) { return generateScalarType(context); },
      /*operand=*/
      [&](const OpaqueContext &context) {
        return generateExpression(context, depth + 1);
      },
      /*constructor=*/
      [](ast::ScalarType &&datatype, ast::Expression &&expression) {
        return ast::CastExpression{std::move(datatype), std::move(expression)};
      });
}

ast::Constant gen::BasicCGenerator::getConstantForType(
    const ast::ScalarType &scalarType) const {
  return llvm::TypeSwitch<ast::ScalarType, ast::Constant>(scalarType)
      .Case([&](const ast::PrimitiveType *primitive) {
        switch (primitive->getType()) {
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
      });
}

std::optional<std::pair<ast::Constant, gen::OpaqueContext>>
gen::BasicCGenerator::generateConstant(const OpaqueContext &context,
                                       std::size_t) const {
  auto candidates = ast::PrimitiveType::ALL_PRIMITIVES;
  random.shuffle(candidates);

  for (ast::PrimitiveType::Type iter : candidates)
    if (std::optional constant =
            typeSystem.discardConstantOpaque(getConstantForType(iter), context))
      return generateWithDependencies<ast::Constant>(
          context, typeSystem.getConstantTransferFns(), *constant);

  return std::nullopt;
}

std::optional<std::pair<ast::ArrayReadExpression, gen::OpaqueContext>>
gen::BasicCGenerator::generateArrayReadExpression(const OpaqueContext &context,
                                                  std::size_t depth) {
  if (typeSystem.discardArrayReadExpressionOpaque(context))
    return std::nullopt;

  return generateWithDependencies<ast::ArrayReadExpression>(
      context, typeSystem.getArrayReadExpressionTransferFns(),
      /*array parameter=*/
      [&](const OpaqueContext &context) {
        return generateArrayParameter(context);
      },
      /*index=*/
      [&](const OpaqueContext &context) {
        return generateExpression(context, depth + 1);
      },
      /*constructor=*/
      [&](ast::ArrayParameter &&param, ast::Expression &&expression) {
        ast::ScalarType elementType = param.getElementType();
        std::size_t mask = param.getDimension() - 1;
        std::string name = param.getName().str();
        // Generate an indexing expression.
        // Has to be an integer.
        ast::Expression index =
            safeCastAsNeeded(ast::PrimitiveType::UInt32, std::move(expression));

        // Bitmask the index to be in range of the array! We use this to avoid
        // undefined behavior in our programs. In the future we could also add
        // mechanisms (type systems, or whatever), that restrict expressions to
        // safe in-range expressions.
        //
        // Note: We can use a bitmask here since array parameters that we
        // generate are all powers-of-2. We do so since the modulo operator is
        // currently unsupported in dynamatic.
        return ast::ArrayReadExpression{
            std::move(elementType), name,
            ast::BinaryExpression{
                std::move(index), ast::BinaryExpression::BitAnd,
                ast::Constant{static_cast<std::uint32_t>(mask)}}};
      });
}

std::optional<std::pair<ast::ArrayParameter, gen::OpaqueContext>>
gen::BasicCGenerator::generateArrayParameter(const OpaqueContext &context,
                                             std::size_t depth) {
  // With a low chance, skip picking an existing parameter and try to generate
  // a new one.
  if (!random.getRatherLowProbabilityBool()) {
    // Randomly shuffle the parameter ordering and find the first parameter
    // that passes type checking.
    std::vector<ast::ArrayParameter> copy = arrayParameters;
    random.shuffle(copy);

    for (const ast::ArrayParameter &candidateParam : copy)
      if (!typeSystem.discardExistingArrayParameterOpaque(candidateParam,
                                                          context))
        return generateWithDependencies<ast::ExistingArrayParameter>(
            context, typeSystem.getExistingArrayParameterTransferFns(),
            candidateParam);
  }

  if (typeSystem.discardFreshArrayParameterOpaque(context))
    return std::nullopt;

  return generateWithDependencies<ast::ArrayParameter>(
      context, typeSystem.getFreshArrayParameterTransferFns(),
      /*element type=*/
      [&](const OpaqueContext &context) { return generateScalarType(context); },
      /*constructor=*/
      [&](ast::ScalarType &&elementType) {
        return arrayParameters.emplace_back(
            std::move(elementType), generateFreshVarName(),
            // Generate a power-of-2 dimension to make the modulo operator
            // fast and easy to implement. We choose an arbitrary upper-bound
            // of 32 for the dimension for now.
            static_cast<std::size_t>(1 << random.getInteger(0, 5)));
      });
}

std::optional<std::pair<ast::Variable, gen::OpaqueContext>>
gen::BasicCGenerator::generateScalarParameter(const OpaqueContext &context,
                                              std::size_t) {
  if (typeSystem.discardVariableOpaque(context))
    return std::nullopt;

  return generateWithDependencies<ast::Variable>(
      context, typeSystem.getVariableTransferFns(),
      /*parameter=*/
      [&](const OpaqueContext &context)
          -> std::optional<std::pair<ast::ScalarParameter, OpaqueContext>> {
        std::array<std::function<std::optional<
                       std::pair<ast::ScalarParameter, OpaqueContext>>()>,
                   2>
            generators;
        generators[0] = [&]()
            -> std::optional<std::pair<ast::ScalarParameter, OpaqueContext>> {
          // Randomly shuffle the parameter ordering and find the first
          // parameter that passes type checking.
          std::vector<ast::ScalarParameter> copy = scalarParameters;
          random.shuffle(copy);

          for (const ast::ScalarParameter &iter : copy)
            if (!typeSystem.discardExistingScalarParameterOpaque(iter, context))
              return generateWithDependencies<ast::ExistingScalarParameter>(
                  context, typeSystem.getExistingScalarParameterTransferFns(),
                  iter);

          return std::nullopt;
        };
        generators[1] = [&]()
            -> std::optional<std::pair<ast::ScalarParameter, OpaqueContext>> {
          if (typeSystem.discardFreshScalarParameterOpaque(context))
            return std::nullopt;

          return generateWithDependencies<ast::ScalarParameter>(
              context, typeSystem.getFreshScalarParameterTransferFns(),
              /*datatype=*/
              [&](const OpaqueContext &context) {
                return generateScalarType(context);
              },
              /*constructor=*/
              [&](ast::ScalarType &&datatype) {
                return scalarParameters.emplace_back(std::move(datatype),
                                                     generateFreshVarName());
              });
        };

        if (random.getRatherLowProbabilityBool())
          std::swap(generators[0], generators[1]);

        for (auto &iter : generators)
          if (std::optional<std::pair<ast::ScalarParameter, OpaqueContext>>
                  result = iter())
            return result;

        return std::nullopt;
      },
      /*constructor=*/
      [&](ast::ScalarParameter &&parameter) {
        return ast::Variable{parameter.getDataType(),
                             parameter.getName().str()};
      });
}

std::optional<std::pair<ast::ScalarType, gen::OpaqueContext>>
gen::BasicCGenerator::generateScalarType(
    const OpaqueContext &context,
    llvm::function_ref<bool(const ast::ScalarType &)> toExclude) const {
  auto candidates = ast::PrimitiveType::ALL_PRIMITIVES;
  random.shuffle(candidates);
  for (const ast::ScalarType &iter : candidates) {
    // Skip some types based on the caller excluding them.
    if (toExclude && toExclude(iter))
      continue;

    if (!typeSystem.discardScalarTypeOpaque(iter, context))
      return generateWithDependencies<ast::ScalarType>(
          context, typeSystem.getScalarTypeTransferFns(), iter);
  }

  return std::nullopt;
}

std::pair<ast::ReturnType, gen::OpaqueContext>
gen::BasicCGenerator::generateReturnType(const OpaqueContext &context) const {
  // Candidates for return types are all primitive types as well as 'void'.
  // (i.e., one more than the number of primitive types).
  std::array<ast::ReturnType, ast::PrimitiveType::ALL_PRIMITIVES.size() + 1>
      candidates;
  llvm::copy(ast::PrimitiveType::ALL_PRIMITIVES, candidates.begin());
  candidates.back() = ast::VoidType{};
  random.shuffle(candidates);
  for (const ast::ReturnType &iter : candidates)
    if (!typeSystem.discardReturnTypeOpaque(iter, context))
      return generateWithDependencies<ast::ReturnType>(
          context, typeSystem.getReturnTypeTransferFns(), iter);

  llvm::report_fatal_error(
      "It must always be possible to generate a return type");
}

constexpr std::size_t MAX_STATEMENTS = 10;

std::pair<ast::StatementList, gen::OpaqueContext>
gen::BasicCGenerator::generateStatementList(const OpaqueContext &context,
                                            size_t depth) {
  if (depth > MAX_STATEMENTS)
    return std::pair{ast::StatementList(), context};

  return generateWithDependencies<ast::StatementList>(
             context, typeSystem.getStatementListTransferFns(),
             /*statement list=*/
             [&](const OpaqueContext &context) {
               return generateStatementList(context, depth + 1);
             },
             /*statement=*/
             [&](const OpaqueContext &context) {
               return generateStatement(context);
             },
             /*constructor=*/
             [&](ast::StatementList &&statements, ast::Statement &&statement) {
               std::vector<ast::Statement> result = statements.takeVector();
               result.push_back(std::move(statement));
               return ast::StatementList(std::move(result));
             })
      .value_or(std::pair{ast::StatementList(), context});
}

std::optional<std::pair<ast::Statement, gen::OpaqueContext>>
gen::BasicCGenerator::generateStatement(const OpaqueContext &context) {
  return generateArrayAssignmentStatement(context);
}

std::optional<std::pair<ast::ArrayAssignmentStatement, gen::OpaqueContext>>
gen::BasicCGenerator::generateArrayAssignmentStatement(
    const OpaqueContext &context) {
  if (typeSystem.discardArrayAssignmentStatementOpaque(context))
    return std::nullopt;

  return generateWithDependencies<ast::ArrayAssignmentStatement>(
      context, typeSystem.getArrayAssignmentStatementTransferFns(),
      /*array parameter=*/
      [&](const OpaqueContext &context) {
        return generateArrayParameter(context);
      },
      /*index=*/
      [&](const OpaqueContext &context) {
        auto [expression, outputContext] =
            generateExpression(context, /*depth=*/0);
        expression = safeCastAsNeeded(
            /*to=*/ast::PrimitiveType::UInt32, std::move(expression));
        return std::pair(std::move(expression), std::move(outputContext));
      },
      /*value=*/
      [&](const OpaqueContext &context) {
        return generateExpression(context, 0);
      },
      /*constructor=*/
      [&](ast::ArrayParameter &&param, ast::Expression &&index,
          ast::Expression &&value) {
        index = ast::BinaryExpression{std::move(index),
                                      ast::BinaryExpression::BitAnd,
                                      ast::Constant{static_cast<std::uint32_t>(
                                          param.getDimension() - 1)}};
        return ast::ArrayAssignmentStatement{
            param.getName().str(),
            std::move(index),
            std::move(value),
        };
      });
}

ast::Function gen::BasicCGenerator::generate(std::string_view functionName) {
  return std::move(
      generateWithDependencies<ast::Function>(
          entryContext, typeSystem.getFunctionTransferFns(),
          /*return type=*/
          [&](const OpaqueContext &context) {
            auto [returnType, outputContext] = generateReturnType(context);
            maybeReturnType = returnType;
            return std::pair{std::move(returnType), std::move(outputContext)};
          },
          /*statement list=*/
          [&](const OpaqueContext &context) {
            return generateStatementList(context, 0);
          },
          /*return statement=*/
          [&](const OpaqueContext &context) {
            return generateReturnStatement(context);
          },
          /*constructor=*/
          [&](ast::ReturnType &&returnType, ast::StatementList &&statements,
              ast::ReturnStatement &&returnStatement) {
            std::optional maybeReturnStatement = std::move(returnStatement);
            if (returnType == ast::VoidType{})
              maybeReturnStatement.reset();

            return ast::Function{
                std::move(returnType),   std::string(functionName),
                scalarParameters,        arrayParameters,
                statements.takeVector(), std::move(maybeReturnStatement),
            };
          })
          ->first);
}

std::string
gen::BasicCGenerator::generateTestBench(const ast::Function &kernel) const {
  std::string s;
  llvm::raw_string_ostream ss(s);
  ss << "\nint main() {\n";
  mlir::raw_indented_ostream os(ss);
  os.indent();
  for (const ast::ScalarParameter &parameter : scalarParameters) {
    os << parameter.getDataType() << ' ' << parameter.getName() << " = "
       << getConstantForType(parameter.getDataType()) << ";\n";
  }

  for (const ast::ArrayParameter &parameter : arrayParameters) {
    os << parameter.getElementType() << ' ' << parameter.getName() << "["
       << parameter.getDimension() << "] = {";
    llvm::interleaveComma(llvm::seq<std::size_t>(0, parameter.getDimension()),
                          os, [&, &parameter = parameter](auto &&) {
                            os << getConstantForType(
                                parameter.getElementType());
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
