#ifndef DYNAMATIC_HLS_FUZZER_LIMITTYPESYSTEM
#define DYNAMATIC_HLS_FUZZER_LIMITTYPESYSTEM

#include "TypeSystem.h"
#include <cstddef>

namespace dynamatic::gen {
struct LimitTypingContext {
  std::size_t expressionDepth{};
  std::size_t totalNumberOfStatements{};
};

/// Typesystem used to enforce limits such as number of a specific AST nodes,
/// parameters, depth of expressions and so on.
class LimitTypeSystem : public TypeSystem<LimitTypingContext, LimitTypeSystem> {

  /// Returns a transfer function which increments 'field' of the context of
  /// 'from' before returning it.
  template <typename ASTNode, std::size_t LimitTypingContext::*field,
            std::size_t from = INPUT_DEPENDENCY>
  static auto incrementDepth() {
    return TransferFn<ASTNode, from>(
        [](LimitTypingContext context, auto &&...) {
          ++(context.*field);
          return context;
        });
  }

public:
  explicit LimitTypeSystem(std::size_t maxExpressionDepth = 4,
                           std::size_t maxTotalStatements = 10)
      : maxExpressionDepth(maxExpressionDepth),
        maxTotalStatements(maxTotalStatements) {}

  bool discardBinaryExpression(ast::BinaryExpression::Op,
                               const LimitTypingContext &context) const {
    return context.expressionDepth >= maxExpressionDepth;
  }

  TransferFnArray<ast::BinaryExpression>
  getBinaryExpressionTransferFns(ast::BinaryExpression::Op op) override {
    return {
        /*lhs=*/incrementDepth<ast::BinaryExpression,
                               &LimitTypingContext::expressionDepth>(),
        /*rhs=*/
        incrementDepth<ast::BinaryExpression,
                       &LimitTypingContext::expressionDepth>(),
        /*output=*/copyInputToOutput<ast::BinaryExpression>(),
    };
  }

  bool discardUnaryExpression(ast::UnaryExpression::Op,
                              const LimitTypingContext &context) const {
    return context.expressionDepth >= maxExpressionDepth;
  }

  TransferFnArray<ast::UnaryExpression>
  getUnaryExpressionTransferFns(ast::UnaryExpression::Op op) override {
    return {
        /*operand=*/incrementDepth<ast::UnaryExpression,
                                   &LimitTypingContext::expressionDepth>(),
        /*output=*/copyInputToOutput<ast::UnaryExpression>(),
    };
  }

  bool discardCastExpression(const LimitTypingContext &context) const {
    return context.expressionDepth >= maxExpressionDepth;
  }

  TransferFnArray<ast::CastExpression> getCastExpressionTransferFns() override {
    return {
        /*target type=*/copyFromInput<ast::CastExpression>(),
        /*operand=*/
        incrementDepth<ast::CastExpression,
                       &LimitTypingContext::expressionDepth>(),
        /*output=*/copyInputToOutput<ast::CastExpression>(),
    };
  }

  bool discardConditionalExpression(const LimitTypingContext &context) const {
    return context.expressionDepth >= maxExpressionDepth;
  }

  TransferFnArray<ast::ConditionalExpression>
  getConditionalExpressionTransferFns() override {
    // Default implementation: Simply propagates the context to the
    // subelements.
    return {
        /*condition=*/incrementDepth<ast::ConditionalExpression,
                                     &LimitTypingContext::expressionDepth>(),
        /*true value=*/
        incrementDepth<ast::ConditionalExpression,
                       &LimitTypingContext::expressionDepth>(),
        /*false value=*/
        incrementDepth<ast::ConditionalExpression,
                       &LimitTypingContext::expressionDepth>(),
        /*output=*/copyInputToOutput<ast::ConditionalExpression>(),
    };
  }

  bool discardArrayReadExpression(const LimitTypingContext &context) const {
    return context.expressionDepth >= maxExpressionDepth;
  }

  TransferFnArray<ast::ArrayReadExpression>
  getArrayReadExpressionTransferFns() override {
    return {
        /*array parameter=*/copyFromInput<ast::ArrayReadExpression>(),
        /*index=*/
        incrementDepth<ast::ArrayReadExpression,
                       &LimitTypingContext::expressionDepth>(),
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
        OutputTransferFn<ast::ArrayAssignmentStatement>(
            std::index_sequence<INPUT_DEPENDENCY>{},
            [](const ast::ArrayAssignmentStatement &,
               LimitTypingContext context) {
              context.totalNumberOfStatements++;
              return context;
            }),
    };
  }

  bool discardStatementList(const LimitTypingContext &context) const {
    return context.totalNumberOfStatements >= maxTotalStatements;
  }

  TransferFnArray<ast::StatementList> getStatementListTransferFns() override {
    // TODO: This needlessly forces in which order statements are to be
    //       generated!
    //       In reality, the type system does not care whether the statement
    //       gets generated first or the rest of the statement list, just that
    //       their respective statement number is propagated to the other.
    //       This is a missing feature!
    return {
        copyFrom<ast::StatementList, ast::StatementList::STATEMENT>(),
        copyFromInput<ast::StatementList>(),
        /*output=*/
        copyToOutput<ast::StatementList, ast::StatementList::STATEMENT_LIST>(),
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
                       &LimitTypingContext::totalNumberOfStatements>(),
        /*output=*/
        copyToOutput<ast::StructuredForStatement,
                     ast::StructuredForStatement::BODY>(),
    };
  }

  static ProbabilityTable<ExpressionKey>
  getExpressionProbabilityTable(const LimitTypingContext &context);

  std::size_t maxExpressionDepth{};
  std::size_t maxTotalStatements{};
};
} // namespace dynamatic::gen

#endif
