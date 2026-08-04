#include "BasicCGenerator.h"

#include "mlir/Support/IndentedOstream.h"

#include "llvm/ADT/ScopeExit.h"

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
gen::BasicCGenerator::generateReturnStatement(OpaqueContext &&context) {
  return *generateWithDependencies<ast::ReturnStatement>(
      std::move(context), typeSystem.getReturnStatementTransferFns(),
      /*return value=*/
      [&](OpaqueContext &&context) {
        auto [expression, outputContext] =
            generateExpression(std::move(context));
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

std::pair<ast::Expression, gen::OpaqueContext>
gen::BasicCGenerator::generateExpression(OpaqueContext &&context) {
  llvm::SmallVector<Constructor<ast::Expression>> constructors = random.shuffle(
      expressionGenerators,
      typeSystem.getExpressionProbabilityTableOpaque(context),
      /*keyF=*/
      &ConstructorKeyPair<ast::Expression,
                          AbstractTypeSystem::ExpressionKey>::second,
      /*mapF=*/
      &ConstructorKeyPair<ast::Expression,
                          AbstractTypeSystem::ExpressionKey>::first);

  // Continuously generate an expression until one passes the type checker.
  for (Constructor<ast::Expression> &con : constructors)
    if (std::optional<std::pair<ast::Expression, OpaqueContext>> result =
            con(this, std::move(context)))
      return std::move(*result);

  llvm_unreachable("it should always be possible to generate an expression");
}

std::optional<std::pair<ast::Expression, gen::OpaqueContext>>
gen::BasicCGenerator::generateBinaryExpression(ast::BinaryExpression::Op op,
                                               OpaqueContext &&context) {
  if (typeSystem.discardBinaryExpressionOpaque(op, context))
    return std::nullopt;

  return generateWithDependencies<ast::BinaryExpression>(
      std::move(context), typeSystem.getBinaryExpressionTransferFns(op),
      /*lhs=*/
      [&](OpaqueContext &&context) {
        return generateExpression(std::move(context));
      },
      /*rhs=*/
      [&](OpaqueContext &&context) {
        return generateExpression(std::move(context));
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
                                              OpaqueContext &&context) {
  if (typeSystem.discardUnaryExpressionOpaque(op, context))
    return std::nullopt;

  return generateWithDependencies<ast::UnaryExpression>(
      std::move(context), typeSystem.getUnaryExpressionTransferFns(op),
      /*operand=*/
      [&](OpaqueContext &&context) {
        return generateExpression(std::move(context));
      },
      /*constructor=*/
      [&](ast::Expression &&operand) -> std::optional<ast::UnaryExpression> {
        return ast::UnaryExpression{op, std::move(operand)};
      });
}

std::optional<std::pair<ast::ConditionalExpression, gen::OpaqueContext>>
gen::BasicCGenerator::generateConditionalExpression(OpaqueContext &&context) {
  if (typeSystem.discardConditionalExpressionOpaque(context))
    return std::nullopt;

  return generateWithDependencies<ast::ConditionalExpression>(
      std::move(context), typeSystem.getConditionalExpressionTransferFns(),
      /*condition=*/
      [&](OpaqueContext &&context) {
        return generateExpression(std::move(context));
      },
      /*true value=*/
      [&](OpaqueContext &&context) {
        return generateExpression(std::move(context));
      },
      /*false value=*/
      [&](OpaqueContext &&context) {
        return generateExpression(std::move(context));
      },
      /*constructor=*/
      [&](ast::Expression &&cond, ast::Expression &&trueExpr,
          ast::Expression &&falseExpr) {
        return ast::ConditionalExpression{std::move(cond), std::move(trueExpr),
                                          std::move(falseExpr)};
      });
}

std::optional<std::pair<ast::CastExpression, gen::OpaqueContext>>
gen::BasicCGenerator::generateCastExpression(OpaqueContext &&context) {
  if (typeSystem.discardCastExpressionOpaque(context))
    return std::nullopt;

  return generateWithDependencies<ast::CastExpression>(
      std::move(context), typeSystem.getCastExpressionTransferFns(),
      /*data type=*/
      [&](OpaqueContext &&context) {
        return generateScalarType(std::move(context));
      },
      /*operand=*/
      [&](OpaqueContext &&context) {
        return generateExpression(std::move(context));
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
gen::BasicCGenerator::generateConstant(OpaqueContext &&context) const {
  auto candidates = ast::PrimitiveType::ALL_PRIMITIVES;
  random.shuffle(candidates);

  for (ast::PrimitiveType::Type iter : candidates)
    if (std::optional constant =
            typeSystem.discardConstantOpaque(getConstantForType(iter), context))
      return generateWithDependencies<ast::Constant>(
          std::move(context), typeSystem.getConstantTransferFns(), *constant);

  return std::nullopt;
}

std::optional<std::pair<ast::ArrayReadExpression, gen::OpaqueContext>>
gen::BasicCGenerator::generateArrayReadExpression(OpaqueContext &&context) {
  if (typeSystem.discardArrayReadExpressionOpaque(context))
    return std::nullopt;

  return generateWithDependencies<ast::ArrayReadExpression>(
      std::move(context), typeSystem.getArrayReadExpressionTransferFns(),
      /*array parameter=*/
      [&](OpaqueContext &&context) {
        return generateArrayParameter(std::move(context));
      },
      /*index=*/
      [&](OpaqueContext &&context) {
        return generateExpression(std::move(context));
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
gen::BasicCGenerator::generateArrayParameter(OpaqueContext &&context) {
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
            std::move(context),
            typeSystem.getExistingArrayParameterTransferFns(), candidateParam);
  }

  if (typeSystem.discardFreshArrayParameterOpaque(context))
    return std::nullopt;

  return generateWithDependencies<ast::ArrayParameter>(
      std::move(context), typeSystem.getFreshArrayParameterTransferFns(),
      /*element type=*/
      [&](OpaqueContext &&context) {
        return generateScalarType(std::move(context));
      },
      /*dimension=*/
      [&](OpaqueContext &&context)
          -> std::optional<std::pair<std::size_t, OpaqueContext>> {
        // Generate a power-of-2 dimension to make the modulo operator fast and
        // easy to implement. We choose an arbitrary upper-bound of 32 for the
        // dimension for now.
        std::size_t dimension = 1 << random.getInteger<std::size_t>(0, 5);
        std::optional<std::size_t> chosen =
            typeSystem.discardArrayDimensionOpaque(dimension, context);
        if (!chosen)
          return std::nullopt;

        return std::pair{*chosen, std::move(context)};
      },
      /*constructor=*/
      [&](ast::ScalarType &&elementType, std::size_t &&dimension) {
        return arrayParameters.emplace_back(std::move(elementType),
                                            generateFreshVarName(), dimension);
      });
}

std::optional<std::pair<ast::Variable, gen::OpaqueContext>>
gen::BasicCGenerator::generateVariable(OpaqueContext &&context) {
  if (typeSystem.discardVariableOpaque(context))
    return std::nullopt;

  return generateWithDependencies<ast::Variable>(
      std::move(context), typeSystem.getVariableTransferFns(),
      /*parameter=*/
      [&](OpaqueContext &&context) {
        return generateScalarParameter(std::move(context));
      },
      /*constructor=*/
      [&](ast::ScalarParameter &&parameter) {
        return ast::Variable{parameter.getDataType(),
                             parameter.getName().str()};
      });
}

std::optional<std::pair<ast::ScalarParameter, gen::OpaqueContext>>
gen::BasicCGenerator::generateScalarParameter(OpaqueContext &&context) {
  std::array<std::function<std::optional<
                 std::pair<ast::ScalarParameter, OpaqueContext>>()>,
             2>
      generators;

  // Existing scalar parameter.
  generators[0] =
      [&]() -> std::optional<std::pair<ast::ScalarParameter, OpaqueContext>> {
    // Randomly shuffle the parameter ordering and find the first
    // parameter that passes type checking.
    std::vector<ast::ScalarParameter> inScope = scalarParameters;
    for (auto &iter : localVariableStack)
      llvm::append_range(inScope, iter);

    llvm::SmallVector<ast::ScalarParameter> candidates = random.shuffle(
        inScope,
        typeSystem.getExistingScalarParameterProbabilityTableOpaque(context),
        /*keyF=*/
        [](const ast::ScalarParameter &parameter) {
          return parameter.getName().str();
        },
        /*mapF=*/llvm::identity<ast::ScalarParameter>{});

    for (const ast::ScalarParameter &iter : candidates)
      if (!typeSystem.discardExistingScalarParameterOpaque(iter, context))
        return generateWithDependencies<ast::ExistingScalarParameter>(
            std::move(context),
            typeSystem.getExistingScalarParameterTransferFns(), iter);

    return std::nullopt;
  };

  // Fresh scalar parameter.
  generators[1] =
      [&]() -> std::optional<std::pair<ast::ScalarParameter, OpaqueContext>> {
    if (typeSystem.discardFreshScalarParameterOpaque(context))
      return std::nullopt;

    return generateWithDependencies<ast::ScalarParameter>(
        std::move(context), typeSystem.getFreshScalarParameterTransferFns(),
        /*datatype=*/
        [&](OpaqueContext &&context) {
          return generateScalarType(std::move(context));
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
    if (std::optional<std::pair<ast::ScalarParameter, OpaqueContext>> result =
            iter())
      return result;

  return std::nullopt;
}

std::optional<std::pair<ast::ScalarType, gen::OpaqueContext>>
gen::BasicCGenerator::generateScalarType(
    OpaqueContext &&context,
    llvm::function_ref<bool(const ast::ScalarType &)> toExclude) const {
  auto candidates = ast::PrimitiveType::ALL_PRIMITIVES;
  random.shuffle(candidates);
  for (const ast::ScalarType &iter : candidates) {
    // Skip some types based on the caller excluding them.
    if (toExclude && toExclude(iter))
      continue;

    if (!typeSystem.discardScalarTypeOpaque(iter, context))
      return generateWithDependencies<ast::ScalarType>(
          std::move(context), typeSystem.getScalarTypeTransferFns(), iter);
  }

  return std::nullopt;
}

std::pair<ast::ReturnType, gen::OpaqueContext>
gen::BasicCGenerator::generateReturnType(OpaqueContext &&context) const {
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
          std::move(context), typeSystem.getReturnTypeTransferFns(), iter);

  llvm::report_fatal_error(
      "It must always be possible to generate a return type");
}

std::pair<ast::StatementList, gen::OpaqueContext>
gen::BasicCGenerator::generateStatementList(OpaqueContext &&context) {
  if (typeSystem.discardStatementListOpaque(context))
    return std::pair{ast::StatementList(), std::move(context)};

  return generateWithDependencies<ast::StatementList>(
             std::move(context), typeSystem.getStatementListTransferFns(),
             /*statement=*/
             [&](OpaqueContext &&context) {
               return generateStatement(std::move(context));
             },
             /*statement list=*/
             [&](OpaqueContext &&context) {
               return generateStatementList(std::move(context));
             },
             /*constructor=*/
             [&](ast::Statement &&statement, ast::StatementList &&statements) {
               // The list is right recursive: the freshly generated
               // statement is the head, prepended before the rest of the
               // list.
               std::vector<ast::Statement> result = statements.takeVector();
               result.insert(result.begin(), std::move(statement));
               return ast::StatementList(std::move(result));
             })
      .value_or(std::pair{ast::StatementList(), std::move(context)});
}

std::optional<std::pair<ast::Statement, gen::OpaqueContext>>
gen::BasicCGenerator::generateStatement(OpaqueContext &&context) {
  llvm::SmallVector<Constructor<ast::Statement>> constructors = random.shuffle(
      statementGenerators,
      typeSystem.getStatementProbabilityTableOpaque(context),
      &ConstructorKeyPair<ast::Statement,
                          AbstractTypeSystem::StatementKey>::second,
      &ConstructorKeyPair<ast::Statement,
                          AbstractTypeSystem::StatementKey>::first);
  for (auto &iter : constructors)
    if (auto result = iter(this, std::move(context)))
      return result;

  return std::nullopt;
}

std::optional<std::pair<ast::ArrayAssignmentStatement, gen::OpaqueContext>>
gen::BasicCGenerator::generateArrayAssignmentStatement(
    OpaqueContext &&context) {
  if (typeSystem.discardArrayAssignmentStatementOpaque(context))
    return std::nullopt;

  return generateWithDependencies<ast::ArrayAssignmentStatement>(
      std::move(context), typeSystem.getArrayAssignmentStatementTransferFns(),
      /*array parameter=*/
      [&](OpaqueContext &&context) {
        return generateArrayParameter(std::move(context));
      },
      /*index=*/
      [&](OpaqueContext &&context) {
        return generateExpression(std::move(context));
      },
      /*value=*/
      [&](OpaqueContext &&context) {
        return generateExpression(std::move(context));
      },
      /*constructor=*/
      [&](ast::ArrayParameter &&param, ast::Expression &&index,
          ast::Expression &&value) {
        index = safeCastAsNeeded(
            /*to=*/ast::PrimitiveType::UInt32, std::move(index));
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

std::optional<std::pair<ast::ScalarAssignmentStatement, gen::OpaqueContext>>
gen::BasicCGenerator::generateScalarAssignmentStatement(
    OpaqueContext &&context) {
  if (typeSystem.discardScalarAssignmentStatementOpaque(context))
    return std::nullopt;

  return generateWithDependencies<ast::ScalarAssignmentStatement>(
      std::move(context), typeSystem.getScalarAssignmentStatementTransferFns(),
      /*target=*/
      [&](OpaqueContext &&context) {
        return generateScalarParameter(std::move(context));
      },
      /*value=*/
      [&](OpaqueContext &&context) {
        return generateExpression(std::move(context));
      },
      /*constructor=*/
      [&](ast::ScalarParameter &&target, ast::Expression &&value) {
        // Ensure the assigned value matches the variable's type without
        // triggering undefined behavior.
        value = safeCastAsNeeded(target.getDataType(), std::move(value));
        return ast::ScalarAssignmentStatement{target.getName().str(),
                                              std::move(value)};
      });
}

std::optional<std::pair<ast::StructuredForStatement, gen::OpaqueContext>>
gen::BasicCGenerator::generateStructuredForStatement(OpaqueContext &&context) {
  if (typeSystem.discardStructuredForStatementOpaque(context))
    return std::nullopt;

  std::string varName = generateFreshVarName();
  return generateWithDependencies<ast::StructuredForStatement>(
      std::move(context), typeSystem.getStructuredForStatementTransferFns(),
      /*iteration variable=*/
      [&](OpaqueContext &&context)
          -> std::optional<std::pair<std::string, OpaqueContext>> {
        // The iteration variable is a terminal whose name is made available to
        // transfer functions (e.g., to forbid writing to it in the body). Its
        // output context is never consumed, so simply forward the input.
        return std::pair{varName, std::move(context)};
      },
      /*start=*/
      [&](OpaqueContext &&context) {
        return generateExpression(std::move(context));
      },
      /*end=*/
      [&](OpaqueContext &&context) {
        return generateExpression(std::move(context));
      },
      /*step=*/
      [&](OpaqueContext &&context) {
        return generateExpression(std::move(context));
      },
      /*body=*/
      [&](OpaqueContext &&context) {
        auto scopeExit = pushNewScope();
        addVariable(ast::PrimitiveType::UInt32, varName);
        return generateStatementList(std::move(context));
      },
      /*constructor=*/
      [&](std::string &&iterVariable, ast::Expression &&start,
          ast::Expression &&end, ast::Expression &&step,
          ast::StatementList &&statements) {
        // Note: Requiring the step to be at least one to ensure termination!
        // TODO: Negative steps are currently unsupported.
        step = generateMaxExpression(step, ast::Constant{1});
        return ast::StructuredForStatement(
            std::move(iterVariable), std::move(start), std::move(end),
            std::move(step), std::move(statements));
      });
}

void gen::BasicCGenerator::initGenerators() {
  expressionGenerators.emplace_back(&BasicCGenerator::generateConstant,
                                    ast::Constant::Tag{});
  expressionGenerators.emplace_back(&BasicCGenerator::generateVariable,
                                    ast::Variable::Tag{});
  for (auto op : enumRange<ast::BinaryExpression::Op>()) {
    expressionGenerators.emplace_back(
        [op](BasicCGenerator *self, OpaqueContext &&context) {
          return self->generateBinaryExpression(op, std::move(context));
        },
        op);
  }
  for (auto op : enumRange<ast::UnaryExpression::Op>()) {
    expressionGenerators.emplace_back(
        [op](BasicCGenerator *self, OpaqueContext &&context) {
          return self->generateUnaryExpression(op, std::move(context));
        },
        op);
  }
  expressionGenerators.emplace_back(&BasicCGenerator::generateCastExpression,
                                    ast::CastExpression::Tag{});
  expressionGenerators.emplace_back(
      &BasicCGenerator::generateArrayReadExpression,
      ast::ArrayReadExpression::Tag{});
  expressionGenerators.emplace_back(
      &BasicCGenerator::generateConditionalExpression,
      ast::ConditionalExpression::Tag{});

  statementGenerators.emplace_back(
      &BasicCGenerator::generateArrayAssignmentStatement,
      ast::ArrayAssignmentStatement::Tag{});
  statementGenerators.emplace_back(
      &BasicCGenerator::generateStructuredForStatement,
      ast::StructuredForStatement::Tag{});
  statementGenerators.emplace_back(
      &BasicCGenerator::generateScalarAssignmentStatement,
      ast::ScalarAssignmentStatement::Tag{});
}

ast::Function gen::BasicCGenerator::generate(llvm::raw_ostream &os,
                                             std::string_view functionName) {
  ast::Function function = generateFunction(functionName);
  os << R"(
#include <stdint.h>
#include <math.h>
#include "dynamatic/Integration.h"

#ifdef HLS_FUZZER_VERIFY
constexpr
#endif
)";
  os << function << '\n';
  os << generateTestBench(function);
  return function;
}

ast::Function
gen::BasicCGenerator::generateFunction(llvm::StringRef functionName) {
  return std::move(
      generateWithDependencies<ast::Function>(
          std::move(entryContext), typeSystem.getFunctionTransferFns(),
          /*return type=*/
          [&](OpaqueContext &&context) {
            auto [returnType, outputContext] =
                generateReturnType(std::move(context));
            maybeReturnType = returnType;
            return std::pair{std::move(returnType), std::move(outputContext)};
          },
          /*statement list=*/
          [&](OpaqueContext &&context) {
            return generateStatementList(std::move(context));
          },
          /*return statement=*/
          [&](OpaqueContext &&context) {
            return generateReturnStatement(std::move(context));
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
  ss << R"(
// Mark the test bench as 'constexpr' to be able to use constant evaluation to
// verify that the kernel is free of undefined-behaviour.
// The 'execute.sh' file in 'TargetUtils.cpp' later appends a 'static_assert'
// that forces constant evaluation of the 'test_bench'.

// Note that we gate this behind the 'HLS_FUZZER_VERIFY' macro since compiling
// the kernel for profiling and I/O generation would otherwise error out as
// things like I/O functions are not 'constexpr' and not allowed to be called
// from 'constexpr' functions in some C++ versions. Furthermore, 'constexpr'
// does not exist in C.
#ifdef HLS_FUZZER_VERIFY
constexpr
#endif
)";
  ss << "void test_bench() {\n";
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
  ss << R"(
int main() {
  test_bench();
}
)";
  return s;
}
