#ifndef DYNAMATIC_HLS_FUZZER_TEMPLATE_TYPE_SYSTEM
#define DYNAMATIC_HLS_FUZZER_TEMPLATE_TYPE_SYSTEM

#include "TypeSystem.h"

namespace dynamatic::gen {

/// CRTP base class implementing every 'get*TransferFns' and 'discard*' method
/// of 'TypeSystem' in terms of three method templates, each of which the
/// derived class may -- but does not have to -- provide:
///
///   template <typename ASTNode, typename... Args>
///   TransferFnArray<ASTNode> getTransferFnImpl(const Args &...args);
///
///   template <typename ASTNode>
///   bool discardTerminalImpl(const ASTNode &node,
///                            const TypingContext &context);
///
///   template <typename ASTNode, typename... Args>
///   bool discardNonTerminalImpl(const TypingContext &context,
///                               const Args &...args);
///
///   template <typename Key>
///   ProbabilityTable<Key> getProbabilityTableImpl(const TypingContext &ctx);
///
/// The class is meant for type systems that treat every AST node the same way
/// and would otherwise have to spell out one identical method per AST node.
/// Whichever of the four a derived class does not define keeps the default
/// below. This is the mirror image of 'TypeSystem::getTransferFn',
/// 'TypeSystem::discardTerminal', 'TypeSystem::discardNonTerminal' and
/// 'TypeSystem::getProbabilityTable'.
///
/// 'ASTNode' is the AST node the method was called about and 'args' are the
/// extra arguments it takes, if any. Probability tables are the exception in
/// that they are dispatched on their 'Key', there being one table per set of
/// AST nodes the generator picks from rather than one per AST node.
///
/// Discarding is split in two because, unlike the transfer function getters,
/// the 'discard*' methods do not all have the same shape:
/// * A *terminal* is an AST node the generator already has in hand and merely
///   asks the type system to accept, i.e. every AST node without subelements.
///   Its method is therefore handed that node on top of the context.
/// * A *non-terminal* is an AST node the generator is about to generate from
///   scratch, i.e. every AST node that has subelements. Its method only ever
///   sees the context plus whatever the generator has already committed to.
///
/// Note that a derived class remains free to implement the method templates
/// for the general case and to define the methods of the individual AST nodes
/// it wants to treat specially, which then take precedence.
///
/// Beware that 'TypeSystem' implements some 'discard*' methods in terms of
/// others and that a 'discardTerminalImpl' replaces all of them at once. A
/// type system that wants that delegation has to implement it within
/// 'discardTerminalImpl' itself.
template <typename TypingContext, typename Self>
class TemplateTypeSystem : public TypeSystem<TypingContext, Self> {
public:
  using Base = TypeSystem<TypingContext, Self>;

  // Default implementations, all of which target 'Base' so that they are
  // exactly what 'TypeSystem' does. Targeting it is what keeps them from
  // coming right back here, this class overriding all of its methods.

  template <typename ASTNode, typename... Args>
  TransferFnArray<ASTNode> getTransferFnImpl(const Args &...args) {
    return Base::template getTransferFn<ASTNode, Base>(args...);
  }

  template <typename ASTNode>
  bool discardTerminalImpl(const ASTNode &node, const TypingContext &context) {
    return Base::template discardTerminal<ASTNode, Base>(node, context);
  }

  template <typename ASTNode, typename... Args>
  bool discardNonTerminalImpl(const TypingContext &context,
                              const Args &...args) {
    return Base::template discardNonTerminal<ASTNode, Base>(context, args...);
  }

  template <typename Key>
  ProbabilityTable<Key> getProbabilityTableImpl(const TypingContext &context) {
    return Base::template getProbabilityTable<Key, Base>(context);
  }

  TransferFnArray<ast::Function> getFunctionTransferFns() override {
    return transferFns<ast::Function>();
  }

  TransferFnArray<ast::ReturnStatement>
  getReturnStatementTransferFns() override {
    return transferFns<ast::ReturnStatement>();
  }

  TransferFnArray<ast::ScalarType> getScalarTypeTransferFns() override {
    return transferFns<ast::ScalarType>();
  }

  TransferFnArray<ast::ReturnType> getReturnTypeTransferFns() override {
    return transferFns<ast::ReturnType>();
  }

  TransferFnArray<ast::BinaryExpression>
  getBinaryExpressionTransferFns(ast::BinaryExpression::Op op) override {
    return transferFns<ast::BinaryExpression>(op);
  }

  TransferFnArray<ast::UnaryExpression>
  getUnaryExpressionTransferFns(ast::UnaryExpression::Op op) override {
    return transferFns<ast::UnaryExpression>(op);
  }

  TransferFnArray<ast::Variable> getVariableTransferFns() override {
    return transferFns<ast::Variable>();
  }

  TransferFnArray<ast::CastExpression> getCastExpressionTransferFns() override {
    return transferFns<ast::CastExpression>();
  }

  TransferFnArray<ast::ConditionalExpression>
  getConditionalExpressionTransferFns() override {
    return transferFns<ast::ConditionalExpression>();
  }

  TransferFnArray<ast::Constant> getConstantTransferFns() override {
    return transferFns<ast::Constant>();
  }

  TransferFnArray<ast::ExistingScalarParameter>
  getExistingScalarParameterTransferFns() override {
    return transferFns<ast::ExistingScalarParameter>();
  }

  TransferFnArray<ast::ScalarParameter>
  getFreshScalarParameterTransferFns() override {
    return transferFns<ast::ScalarParameter>();
  }

  TransferFnArray<ast::ArrayReadExpression>
  getArrayReadExpressionTransferFns() override {
    return transferFns<ast::ArrayReadExpression>();
  }

  TransferFnArray<ast::ExistingArrayParameter>
  getExistingArrayParameterTransferFns() override {
    return transferFns<ast::ExistingArrayParameter>();
  }

  TransferFnArray<ast::ArrayParameter>
  getFreshArrayParameterTransferFns() override {
    return transferFns<ast::ArrayParameter>();
  }

  TransferFnArray<ast::ArrayAssignmentStatement>
  getArrayAssignmentStatementTransferFns() override {
    return transferFns<ast::ArrayAssignmentStatement>();
  }

  TransferFnArray<ast::ScalarAssignmentStatement>
  getScalarAssignmentStatementTransferFns() override {
    return transferFns<ast::ScalarAssignmentStatement>();
  }

  TransferFnArray<ast::StatementList> getStatementListTransferFns() override {
    return transferFns<ast::StatementList>();
  }

  TransferFnArray<ast::StructuredForStatement>
  getStructuredForStatementTransferFns() override {
    return transferFns<ast::StructuredForStatement>();
  }

  bool discardScalarType(const ast::ScalarType &scalarType,
                         const TypingContext &context) {
    return terminal(scalarType, context);
  }

  bool discardReturnType(const ast::ReturnType &returnType,
                         const TypingContext &context) {
    return terminal(returnType, context);
  }

  /// Note that a constant is only ever kept or discarded here. A type system
  /// wanting to replace it with another one has to define this method itself.
  std::optional<ast::Constant> discardConstant(const ast::Constant &constant,
                                               const TypingContext &context) {
    if (terminal(constant, context))
      return std::nullopt;
    return constant;
  }

  /// Note that a dimension is only ever kept or discarded here. A type system
  /// wanting to replace it with another one has to define this method itself.
  std::optional<std::size_t>
  discardArrayDimension(std::size_t dimension, const TypingContext &context) {
    if (terminal(dimension, context))
      return std::nullopt;
    return dimension;
  }

  bool discardExistingScalarParameter(const ast::ScalarParameter &parameter,
                                      const TypingContext &context) {
    return terminal(parameter, context);
  }

  bool discardExistingArrayParameter(const ast::ArrayParameter &parameter,
                                     const TypingContext &context) {
    return terminal(parameter, context);
  }

  bool discardBinaryExpression(ast::BinaryExpression::Op op,
                               const TypingContext &context) {
    return nonTerminal<ast::BinaryExpression>(context, op);
  }

  bool discardUnaryExpression(ast::UnaryExpression::Op op,
                              const TypingContext &context) {
    return nonTerminal<ast::UnaryExpression>(context, op);
  }

  bool discardVariable(const TypingContext &context) {
    return nonTerminal<ast::Variable>(context);
  }

  bool discardCastExpression(const TypingContext &context) {
    return nonTerminal<ast::CastExpression>(context);
  }

  bool discardConditionalExpression(const TypingContext &context) {
    return nonTerminal<ast::ConditionalExpression>(context);
  }

  bool discardFreshScalarParameter(const TypingContext &context) {
    return nonTerminal<ast::ScalarParameter>(context);
  }

  bool discardArrayReadExpression(const TypingContext &context) {
    return nonTerminal<ast::ArrayReadExpression>(context);
  }

  bool discardFreshArrayParameter(const TypingContext &context) {
    return nonTerminal<ast::ArrayParameter>(context);
  }

  bool discardArrayAssignmentStatement(const TypingContext &context) {
    return nonTerminal<ast::ArrayAssignmentStatement>(context);
  }

  bool discardScalarAssignmentStatement(const TypingContext &context) {
    return nonTerminal<ast::ScalarAssignmentStatement>(context);
  }

  bool discardStatementList(const TypingContext &context) {
    return nonTerminal<ast::StatementList>(context);
  }

  bool discardStructuredForStatement(const TypingContext &context) {
    return nonTerminal<ast::StructuredForStatement>(context);
  }

  ProbabilityTable<AbstractTypeSystem::ExpressionKey>
  getExpressionProbabilityTable(const TypingContext &context) {
    return probabilityTable<AbstractTypeSystem::ExpressionKey>(context);
  }

  ProbabilityTable<AbstractTypeSystem::StatementKey>
  getStatementProbabilityTable(const TypingContext &context) {
    return probabilityTable<AbstractTypeSystem::StatementKey>(context);
  }

  ProbabilityTable<AbstractTypeSystem::ExistingScalarParameterKey>
  getExistingScalarParameterProbabilityTable(const TypingContext &context) {
    return probabilityTable<AbstractTypeSystem::ExistingScalarParameterKey>(
        context);
  }

private:
  /// Calls the derived class' 'getTransferFnImpl' for 'ASTNode'.
  template <typename ASTNode, typename... Args>
  TransferFnArray<ASTNode> transferFns(const Args &...args) {
    return self().template getTransferFnImpl<ASTNode>(args...);
  }

  /// Calls the derived class' 'discardTerminalImpl' for 'node'.
  template <typename ASTNode>
  bool terminal(const ASTNode &node, const TypingContext &context) {
    return self().discardTerminalImpl(node, context);
  }

  /// Calls the derived class' 'discardNonTerminalImpl' for 'ASTNode'.
  template <typename ASTNode, typename... Args>
  bool nonTerminal(const TypingContext &context, const Args &...args) {
    return self().template discardNonTerminalImpl<ASTNode>(context, args...);
  }

  /// Calls the derived class' 'getProbabilityTableImpl' for 'Key'.
  template <typename Key>
  ProbabilityTable<Key> probabilityTable(const TypingContext &context) {
    return self().template getProbabilityTableImpl<Key>(context);
  }

  Self &self() { return static_cast<Self &>(*this); }
};

/// Convenience type system that disallows every AST constructs (besides
/// functions) by default.
template <typename TypingContext, typename Self>
class DisallowByDefaultTypeSystem
    : public TemplateTypeSystem<TypingContext, Self> {
public:
  template <typename ASTNode>
  static bool discardTerminalImpl(const ASTNode &, const TypingContext &) {
    return true;
  }

  template <typename ASTNode, typename... Args>
  static bool discardNonTerminalImpl(const TypingContext &, const Args &...) {
    return true;
  }
};

} // namespace dynamatic::gen

#endif
