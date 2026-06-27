#ifndef DYNAMATIC_HLS_FUZZER_LIMITTYPESYSTEM
#define DYNAMATIC_HLS_FUZZER_LIMITTYPESYSTEM

#include "ConjunctionTypeSystem.h"
#include "CounterTypeSystem.h"
#include "TypeSystem.h"
#include <cstddef>

namespace dynamatic::gen {

// Internal implementation details. Users should use the LimitTypeSystem
// instead.
namespace details {

struct DepthTypingContext {
  /// The current expression depth.
  std::size_t expressionDepth{};
  /// The current number of statements in the program.
  std::size_t totalNumberOfStatements{};
};

class DepthTypeSystem : public TypeSystem<DepthTypingContext, DepthTypeSystem> {

  /// Returns a transfer function which increments 'field' of the context of
  /// 'from' before returning it.
  template <typename ASTNode, std::size_t DepthTypingContext::*field,
            std::size_t from = INPUT_DEPENDENCY>
  static auto incrementDepth() {
    return TransferFn<ASTNode, from>(
        [](DepthTypingContext context, auto &&...) {
          ++(context.*field);
          return context;
        });
  }

public:
  explicit DepthTypeSystem(std::size_t maxExpressionDepth,
                           std::size_t maxTotalStatements)
      : maxExpressionDepth(maxExpressionDepth),
        maxTotalStatements(maxTotalStatements) {}

  bool discardBinaryExpression(ast::BinaryExpression::Op,
                               const DepthTypingContext &context) const {
    return context.expressionDepth >= maxExpressionDepth;
  }

  TransferFnArray<ast::BinaryExpression>
  getBinaryExpressionTransferFns(ast::BinaryExpression::Op op) override {
    return {
        /*lhs=*/incrementDepth<ast::BinaryExpression,
                               &DepthTypingContext::expressionDepth>(),
        /*rhs=*/
        incrementDepth<ast::BinaryExpression,
                       &DepthTypingContext::expressionDepth>(),
        /*output=*/copyInputToOutput<ast::BinaryExpression>(),
    };
  }

  bool discardUnaryExpression(ast::UnaryExpression::Op,
                              const DepthTypingContext &context) const {
    return context.expressionDepth >= maxExpressionDepth;
  }

  TransferFnArray<ast::UnaryExpression>
  getUnaryExpressionTransferFns(ast::UnaryExpression::Op op) override {
    return {
        /*operand=*/incrementDepth<ast::UnaryExpression,
                                   &DepthTypingContext::expressionDepth>(),
        /*output=*/copyInputToOutput<ast::UnaryExpression>(),
    };
  }

  bool discardCastExpression(const DepthTypingContext &context) const {
    return context.expressionDepth >= maxExpressionDepth;
  }

  TransferFnArray<ast::CastExpression> getCastExpressionTransferFns() override {
    return {
        /*target type=*/copyFromInput<ast::CastExpression>(),
        /*operand=*/
        incrementDepth<ast::CastExpression,
                       &DepthTypingContext::expressionDepth>(),
        /*output=*/copyInputToOutput<ast::CastExpression>(),
    };
  }

  bool discardConditionalExpression(const DepthTypingContext &context) const {
    return context.expressionDepth >= maxExpressionDepth;
  }

  TransferFnArray<ast::ConditionalExpression>
  getConditionalExpressionTransferFns() override {
    // Default implementation: Simply propagates the context to the
    // subelements.
    return {
        /*condition=*/incrementDepth<ast::ConditionalExpression,
                                     &DepthTypingContext::expressionDepth>(),
        /*true value=*/
        incrementDepth<ast::ConditionalExpression,
                       &DepthTypingContext::expressionDepth>(),
        /*false value=*/
        incrementDepth<ast::ConditionalExpression,
                       &DepthTypingContext::expressionDepth>(),
        /*output=*/copyInputToOutput<ast::ConditionalExpression>(),
    };
  }

  bool discardArrayReadExpression(const DepthTypingContext &context) const {
    return context.expressionDepth >= maxExpressionDepth;
  }

  TransferFnArray<ast::ArrayReadExpression>
  getArrayReadExpressionTransferFns() override {
    return {
        /*array parameter=*/copyFromInput<ast::ArrayReadExpression>(),
        /*index=*/
        incrementDepth<ast::ArrayReadExpression,
                       &DepthTypingContext::expressionDepth>(),
        /*output=*/copyInputToOutput<ast::ArrayReadExpression>(),
    };
  }

  TransferFnArray<ast::ArrayAssignmentStatement>
  getArrayAssignmentStatementTransferFns() override {
    return TransferFnArray<ast::ArrayAssignmentStatement>{
        /*array parameter=*/copyFromInput<ast::ArrayAssignmentStatement>(),
        /*index=*/copyFromInput<ast::ArrayAssignmentStatement>(),
        /*value=*/copyFromInput<ast::ArrayAssignmentStatement>(),
        /*output=*/
        OutputTransferFn<ast::ArrayAssignmentStatement, INPUT_DEPENDENCY>(
            [](const ast::ArrayAssignmentStatement &,
               DepthTypingContext context) {
              context.totalNumberOfStatements++;
              return context;
            }),
    };
  }

  TransferFnArray<ast::ScalarAssignmentStatement>
  getScalarAssignmentStatementTransferFns() override {
    return TransferFnArray<ast::ScalarAssignmentStatement>{
        /*target=*/copyFromInput<ast::ScalarAssignmentStatement>(),
        /*value=*/copyFromInput<ast::ScalarAssignmentStatement>(),
        /*output=*/
        OutputTransferFn<ast::ScalarAssignmentStatement, INPUT_DEPENDENCY>(
            [](const ast::ScalarAssignmentStatement &,
               DepthTypingContext context) {
              context.totalNumberOfStatements++;
              return context;
            }),
    };
  }

  bool discardStatementList(const DepthTypingContext &context) const {
    return context.totalNumberOfStatements >= maxTotalStatements;
  }

  TransferFnArray<ast::StatementList> getStatementListTransferFns() override {
    return {
        /*statement list=*/copyFirstOf<ast::StatementList,
                                       weak(ast::StatementList::STATEMENT),
                                       INPUT_DEPENDENCY>(),
        /*statement=*/
        copyFirstOf<ast::StatementList,
                    weak(ast::StatementList::STATEMENT_LIST),
                    INPUT_DEPENDENCY>(),
        /*output=*/
        OutputTransferFn<ast::StatementList, ast::StatementList::STATEMENT,
                         ast::StatementList::STATEMENT_LIST>(
            [](const auto &, DepthTypingContext statement,
               const DepthTypingContext &statementList) {
              // Regardless of which of the two was generated first, we can
              // extract the total number of statements by taking their maximum.
              statement.totalNumberOfStatements =
                  std::max(statement.totalNumberOfStatements,
                           statementList.totalNumberOfStatements);
              return statement;
            }),
    };
  }

  TransferFnArray<ast::StructuredForStatement>
  getStructuredForStatementTransferFns() override {
    return {
        /*start=*/copyFromInput<ast::StructuredForStatement>(),
        /*end=*/copyFromInput<ast::StructuredForStatement>(),
        /*step=*/copyFromInput<ast::StructuredForStatement>(),
        /*statements=*/
        incrementDepth<ast::StructuredForStatement,
                       &DepthTypingContext::totalNumberOfStatements>(),
        /*output=*/
        copyToOutput<ast::StructuredForStatement,
                     ast::StructuredForStatement::BODY>(),
    };
  }

  static ProbabilityTable<ExpressionKey>
  getExpressionProbabilityTable(const DepthTypingContext &context);

  std::size_t maxExpressionDepth{};
  std::size_t maxTotalStatements{};
};

struct ParamTypingContext {
  std::size_t numScalarParam{};
  std::size_t numArrayParam{};

  ParamTypingContext merge(const ParamTypingContext &rhs) const {
    return {std::max(numScalarParam, rhs.numScalarParam),
            std::max(numArrayParam, rhs.numArrayParam)};
  }
};

/// Type system that caps the maximum amount of scalar parameters, array
/// parameters and parameters in general.
class ParamTypeSystem
    : public CounterTypeSystem<ParamTypingContext, ParamTypeSystem> {
public:
  explicit ParamTypeSystem(std::size_t maxScalarParam,
                           std::size_t maxArrayParam, std::size_t maxParams)
      : maxScalarParam(maxScalarParam), maxArrayParam(maxArrayParam),
        maxParams(maxParams) {}

  bool discardFreshScalarParameter(const ParamTypingContext &context) const {
    return context.numScalarParam >= maxScalarParam ||
           context.numScalarParam + context.numArrayParam >= maxParams;
  }

  TransferFnArray<ast::ScalarParameter>
  getFreshScalarParameterTransferFns() override {
    return {
        copyFromInput<ast::ScalarParameter>(),
        OutputTransferFn<ast::ScalarParameter, INPUT_DEPENDENCY>(
            [](const ast::ScalarParameter &, ParamTypingContext context) {
              context.numScalarParam++;
              return context;
            }),
    };
  }

  bool discardFreshArrayParameter(const ParamTypingContext &context) const {
    return context.numArrayParam >= maxArrayParam ||
           context.numScalarParam + context.numArrayParam >= maxParams;
  }

  TransferFnArray<ast::ArrayParameter>
  getFreshArrayParameterTransferFns() override {
    return {
        copyFromInput<ast::ArrayParameter>(),
        OutputTransferFn<ast::ArrayParameter, INPUT_DEPENDENCY>(
            [](const ast::ArrayParameter &, ParamTypingContext context) {
              context.numArrayParam++;
              return context;
            }),
    };
  }

private:
  std::size_t maxScalarParam{};
  std::size_t maxArrayParam{};
  std::size_t maxParams{};
};

} // namespace details

/// Typesystem used to enforce limits such as number of a specific AST nodes,
/// parameters, depth of expressions and so on.
class LimitTypeSystem final
    : public ConjunctionTypeSystemBase<
          LimitTypeSystem, details::DepthTypeSystem, details::ParamTypeSystem> {
public:
  struct Options {
    Options() = default;

    std::size_t maxExpressionDepth = 4;
    std::size_t maxTotalStatements = 10;
    std::size_t maxScalarParam = 16;
    std::size_t maxArrayParam = 8;
    std::size_t maxParams = 256;
  };

  LimitTypeSystem() : LimitTypeSystem(Options()) {}

  explicit LimitTypeSystem(const Options &options)
      : ConjunctionTypeSystemBase(
            details::DepthTypeSystem(options.maxExpressionDepth,
                                     options.maxTotalStatements),
            details::ParamTypeSystem(options.maxScalarParam,
                                     options.maxArrayParam,
                                     options.maxParams)) {}
};
} // namespace dynamatic::gen

#endif
