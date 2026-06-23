#ifndef DYNAMATIC_HLS_FUZZER_VISITOR_TYPE_SYSTEM
#define DYNAMATIC_HLS_FUZZER_VISITOR_TYPE_SYSTEM

#include "TypeSystem.h"

namespace dynamatic::gen {

/// Convenient base class for any type systems that are "pure" visitors.
/// These type systems:
/// * Do not care about the order that AST nodes are generated.
/// * Have a monotonic 'TypingContext' which only ever increases/decreases.
///
/// One property of this type system is that the most recent transfer function
/// called is guaranteed to receive the maximum/minimum 'TypingContext'
/// instance.
/// This makes the type system especially useful to implement counters and
/// similar.
///
/// The 'TypingContext' is required to contain a method with the signature:
/// 'TypingContext merge(const TypingContext& rhs) const' which can be used
/// to calculate the current maximum/minimum of all contexts generated so far.
template <typename TypingContext, typename Self>
class VisitorTypeSystem : public TypeSystem<TypingContext, Self> {

public:
  using Base = TypeSystem<TypingContext, Self>;

protected:
  /// Returns a 'TransferFn' which merges all present contexts of 'indices' and
  /// the input context.
  template <typename ASTNode, std::size_t... indices>
  static auto getMergingTransferFn(std::index_sequence<indices...>) {
    return TransferFn<TypingContext, ASTNode, INPUT_DEPENDENCY,
                      weak(indices)...>(
        [](TypingContext result, auto &&...args) -> TypingContext {
          foreachInTuples(
              [&](const auto &element) {
                if constexpr (std::is_same_v<std::decay_t<decltype(element)>,
                                             const TypingContext *>) {
                  if (!element)
                    return;
                  result = result.merge(*element);
                }
              },
              std::forward_as_tuple(std::forward<decltype(args)>(args)...));
          return result;
        });
  }

  /// Returns a 'TransferFn' which merges all present contexts including
  /// the input context.
  template <typename ASTNode>
  static auto getMergingTransferFn() {
    return getMergingTransferFn<ASTNode>(
        std::make_index_sequence<
            std::tuple_size_v<typename ASTNode::SubElements>>{});
  }

  /// Returns a 'OutputTransferFn' which merges all present contexts of
  /// 'indices' and the input context.
  template <typename ASTNode, std::size_t... indices>
  static auto getMergingOutputTransferFn(std::index_sequence<indices...>) {
    return OutputTransferFn<TypingContext, ASTNode, INPUT_DEPENDENCY,
                            indices...>(
        [](const ASTNode &, TypingContext result,
           const auto &...args) -> TypingContext {
          foreachInTuples(
              [&](const auto &element) {
                if constexpr (std::is_same_v<std::decay_t<decltype(element)>,
                                             TypingContext>) {
                  result = result.merge(element);
                }
              },
              std::forward_as_tuple(std::forward<decltype(args)>(args)...));
          return result;
        });
  }

  /// Returns a 'OutputTransferFn' which merges all present contexts including
  /// the input context.
  template <typename ASTNode>
  static auto getMergingOutputTransferFn() {
    return getMergingOutputTransferFn<ASTNode>(
        std::make_index_sequence<
            std::tuple_size_v<typename ASTNode::SubElements>>{});
  }

  /// Returns a transfer array only consisting of merging transfer functions
  /// and output functions.
  template <typename ASTNode>
  static TransferFnArray<ASTNode> getMergingTransferFnArray() {
    return std::tuple_cat(
        mapTuples([](auto &&) { return getMergingTransferFn<ASTNode>(); },
                  getTupleOfIndices(
                      std::make_index_sequence<
                          std::tuple_size_v<typename ASTNode::SubElements>>{})),
        std::tuple(getMergingOutputTransferFn<ASTNode>()));
  }

public:
  TransferFnArray<ast::Function> getFunctionTransferFns() override {
    return getMergingTransferFnArray<ast::Function>();
  }

  TransferFnArray<ast::ScalarType> getScalarTypeTransferFns() override {
    return getMergingTransferFnArray<ast::ScalarType>();
  }

  TransferFnArray<ast::ReturnType> getReturnTypeTransferFns() override {
    return getMergingTransferFnArray<ast::ReturnType>();
  }

  TransferFnArray<ast::ReturnStatement>
  getReturnStatementTransferFns() override {
    return getMergingTransferFnArray<ast::ReturnStatement>();
  }

  TransferFnArray<ast::BinaryExpression>
  getBinaryExpressionTransferFns(ast::BinaryExpression::Op op) override {
    return getMergingTransferFnArray<ast::BinaryExpression>();
  }

  TransferFnArray<ast::UnaryExpression>
  getUnaryExpressionTransferFns(ast::UnaryExpression::Op op) override {
    return getMergingTransferFnArray<ast::UnaryExpression>();
  }

  TransferFnArray<ast::Variable> getVariableTransferFns() override {
    return getMergingTransferFnArray<ast::Variable>();
  }

  TransferFnArray<ast::CastExpression> getCastExpressionTransferFns() override {
    return getMergingTransferFnArray<ast::CastExpression>();
  }

  TransferFnArray<ast::ConditionalExpression>
  getConditionalExpressionTransferFns() override {
    return getMergingTransferFnArray<ast::ConditionalExpression>();
  }

  TransferFnArray<ast::Constant> getConstantTransferFns() override {
    return getMergingTransferFnArray<ast::Constant>();
  }

  TransferFnArray<ast::ScalarParameter>
  getFreshScalarParameterTransferFns() override {
    return getMergingTransferFnArray<ast::ScalarParameter>();
  }

  TransferFnArray<ast::ExistingScalarParameter>
  getExistingScalarParameterTransferFns() override {
    return getMergingTransferFnArray<ast::ExistingScalarParameter>();
  }

  TransferFnArray<ast::ArrayReadExpression>
  getArrayReadExpressionTransferFns() override {
    return getMergingTransferFnArray<ast::ArrayReadExpression>();
  }

  TransferFnArray<ast::ArrayParameter>
  getFreshArrayParameterTransferFns() override {
    return getMergingTransferFnArray<ast::ArrayParameter>();
  }

  TransferFnArray<ast::ExistingArrayParameter>
  getExistingArrayParameterTransferFns() override {
    return getMergingTransferFnArray<ast::ExistingArrayParameter>();
  }

  TransferFnArray<ast::ArrayAssignmentStatement>
  getArrayAssignmentStatementTransferFns() override {
    return getMergingTransferFnArray<ast::ArrayAssignmentStatement>();
  }

  TransferFnArray<ast::StatementList> getStatementListTransferFns() override {
    return getMergingTransferFnArray<ast::StatementList>();
  }

  TransferFnArray<ast::StructuredForStatement>
  getStructuredForStatementTransferFns() override {
    return getMergingTransferFnArray<ast::StructuredForStatement>();
  }
};

} // namespace dynamatic::gen

#endif
