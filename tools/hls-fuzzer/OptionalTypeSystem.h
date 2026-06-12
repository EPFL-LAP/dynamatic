#ifndef DYNAMATIC_HLS_FUZZER_OPTIONALTYPESYSTEM
#define DYNAMATIC_HLS_FUZZER_OPTIONALTYPESYSTEM

#include "TypeSystem.h"
#include <optional>

namespace dynamatic::gen {

/// Wrapper around a 'SubTypeSystem' that allows one to selectively enable
/// the type system as needed.
/// To enable the type system the 'enableTypeSystemFor' function should be
/// used.
/// It allows enabling the type system for specific sub-elements of an
/// 'ASTNode' by setting their input contexts to a given entryContext.
///
/// An 'OptionalTypeSystem' is therefore best used within a
/// 'ConjunctionTypeSystem' or similar, where another type systems logic can
/// enable sub-trees.
///
/// Otherwise, if the context is an empty optional, and the type system
/// therefore disabled, it acts identical to a 'NoopTypeSystem'.
template <typename SubTypeSystem>
class OptionalTypeSystem final
    : public TypeSystem<std::optional<typename SubTypeSystem::Context>,
                        OptionalTypeSystem<SubTypeSystem>> {
public:
  using SubContext = typename SubTypeSystem::Context;
  using Base = TypeSystem<std::optional<SubContext>, OptionalTypeSystem>;
  using Context = typename Base::Context;

  /*implicit*/ OptionalTypeSystem(SubTypeSystem &&subTypeSystem)
      : subTypeSystem(std::move(subTypeSystem)) {}

  /// Enables the sub type system for specific 'subElementsToEnable' of
  /// 'ASTNode'.
  /// The input context given to these sub-elements is 'entryContext'.
  /// It is the users responsibility to make sure that the given 'entryContext'
  /// makes sense for the sub type system.
  template <typename ASTNode, std::size_t... subElementsToEnable>
  static TransferFnArray<ASTNode>
  enableTypeSystemFor(TransferFnArray<ASTNode> transferFnArray,
                      const SubContext &entryContext) {
    ((std::get<subElementsToEnable>(transferFnArray) =
          typename Base::template TransferFn<ASTNode>(entryContext)),
     ...);
    return transferFnArray;
  }

  TransferFnArray<ast::Function> getFunctionTransferFns() override {
    return wrapTransferFns<ast::Function>(
        subTypeSystem.getFunctionTransferFns());
  }

  TransferFnArray<ast::ReturnStatement>
  getReturnStatementTransferFns() override {
    return wrapTransferFns<ast::ReturnStatement>(
        subTypeSystem.getReturnStatementTransferFns());
  }

  bool discardScalarType(const ast::ScalarType &scalarType,
                         const Context &context) {
    if (!context)
      return false;
    return subTypeSystem.discardScalarType(scalarType, *context);
  }

  TransferFnArray<ast::ScalarType> getScalarTypeTransferFns() override {
    return wrapTransferFns<ast::ScalarType>(
        subTypeSystem.getScalarTypeTransferFns());
  }

  bool discardReturnType(const ast::ReturnType &returnType,
                         const Context &context) {
    if (!context)
      return false;
    return subTypeSystem.discardReturnType(returnType, *context);
  }

  TransferFnArray<ast::ReturnType> getReturnTypeTransferFns() override {
    return wrapTransferFns<ast::ReturnType>(
        subTypeSystem.getReturnTypeTransferFns());
  }

  bool discardBinaryExpression(ast::BinaryExpression::Op op,
                               const Context &context) {
    if (!context)
      return false;
    return subTypeSystem.discardBinaryExpression(op, *context);
  }

  TransferFnArray<ast::BinaryExpression>
  getBinaryExpressionTransferFns(ast::BinaryExpression::Op op) override {
    return wrapTransferFns<ast::BinaryExpression>(
        subTypeSystem.getBinaryExpressionTransferFns(op));
  }

  bool discardUnaryExpression(ast::UnaryExpression::Op op,
                              const Context &context) {
    if (!context)
      return false;
    return subTypeSystem.discardUnaryExpression(op, *context);
  }

  TransferFnArray<ast::UnaryExpression>
  getUnaryExpressionTransferFns(ast::UnaryExpression::Op op) override {
    return wrapTransferFns<ast::UnaryExpression>(
        subTypeSystem.getUnaryExpressionTransferFns(op));
  }

  bool discardVariable(const Context &context) {
    if (!context)
      return false;
    return subTypeSystem.discardVariable(*context);
  }

  TransferFnArray<ast::Variable> getVariableTransferFns() override {
    return wrapTransferFns<ast::Variable>(
        subTypeSystem.getVariableTransferFns());
  }

  bool discardCastExpression(const Context &context) {
    if (!context)
      return false;
    return subTypeSystem.discardCastExpression(*context);
  }

  TransferFnArray<ast::CastExpression> getCastExpressionTransferFns() override {
    return wrapTransferFns<ast::CastExpression>(
        subTypeSystem.getCastExpressionTransferFns());
  }

  bool discardConditionalExpression(const Context &context) {
    if (!context)
      return false;
    return subTypeSystem.discardConditionalExpression(*context);
  }

  TransferFnArray<ast::ConditionalExpression>
  getConditionalExpressionTransferFns() override {
    return wrapTransferFns<ast::ConditionalExpression>(
        subTypeSystem.getConditionalExpressionTransferFns());
  }

  std::optional<ast::Constant> discardConstant(const ast::Constant &constant,
                                               const Context &context) {
    if (!context)
      return constant;
    return subTypeSystem.discardConstant(constant, *context);
  }

  TransferFnArray<ast::Constant> getConstantTransferFns() override {
    return wrapTransferFns<ast::Constant>(
        subTypeSystem.getConstantTransferFns());
  }

  bool discardExistingScalarParameter(const ast::ScalarParameter &parameter,
                                      const Context &context) {
    if (!context)
      return false;
    return subTypeSystem.discardExistingScalarParameter(parameter, *context);
  }

  TransferFnArray<ast::ExistingScalarParameter>
  getExistingScalarParameterTransferFns() override {
    return wrapTransferFns<ast::ExistingScalarParameter>(
        subTypeSystem.getExistingScalarParameterTransferFns());
  }

  bool discardFreshScalarParameter(const Context &context) {
    if (!context)
      return false;
    return subTypeSystem.discardFreshScalarParameter(*context);
  }

  TransferFnArray<ast::ScalarParameter>
  getFreshScalarParameterTransferFns() override {
    return wrapTransferFns<ast::ScalarParameter>(
        subTypeSystem.getFreshScalarParameterTransferFns());
  }

  bool discardArrayReadExpression(const Context &context) {
    if (!context)
      return false;
    return subTypeSystem.discardArrayReadExpression(*context);
  }

  TransferFnArray<ast::ArrayReadExpression>
  getArrayReadExpressionTransferFns() override {
    return wrapTransferFns<ast::ArrayReadExpression>(
        subTypeSystem.getArrayReadExpressionTransferFns());
  }

  bool discardExistingArrayParameter(const ast::ArrayParameter &parameter,
                                     const Context &context) {
    if (!context)
      return false;
    return subTypeSystem.discardExistingArrayParameter(parameter, *context);
  }

  TransferFnArray<ast::ExistingArrayParameter>
  getExistingArrayParameterTransferFns() override {
    return wrapTransferFns<ast::ExistingArrayParameter>(
        subTypeSystem.getExistingArrayParameterTransferFns());
  }

  bool discardFreshArrayParameter(const Context &context) {
    if (!context)
      return false;
    return subTypeSystem.discardFreshArrayParameter(*context);
  }

  TransferFnArray<ast::ArrayParameter>
  getFreshArrayParameterTransferFns() override {
    return wrapTransferFns<ast::ArrayParameter>(
        subTypeSystem.getFreshArrayParameterTransferFns());
  }

  bool discardArrayAssignmentStatement(const Context &context) {
    if (!context)
      return false;
    return subTypeSystem.discardArrayAssignmentStatement(*context);
  }

  TransferFnArray<ast::ArrayAssignmentStatement>
  getArrayAssignmentStatementTransferFns() override {
    return wrapTransferFns<ast::ArrayAssignmentStatement>(
        subTypeSystem.getArrayAssignmentStatementTransferFns());
  }

  bool discardStatementList(const Context &context) {
    if (!context)
      return false;
    return subTypeSystem.discardStatementList(*context);
  }

  TransferFnArray<ast::StatementList> getStatementListTransferFns() override {
    return wrapTransferFns<ast::StatementList>(
        subTypeSystem.getStatementListTransferFns());
  }

  bool discardStructuredForStatement(const Context &context) {
    if (!context)
      return false;
    return subTypeSystem.discardStructuredForStatement(*context);
  }

  TransferFnArray<ast::StructuredForStatement>
  getStructuredForStatementTransferFns() override {
    return wrapTransferFns<ast::StructuredForStatement>(
        subTypeSystem.getStructuredForStatementTransferFns());
  }

  ProbabilityTable<AbstractTypeSystem::ExpressionKey>
  getExpressionProbabilityTable(const Context &context) {
    if (!context)
      return Base::getExpressionProbabilityTable(context);

    return subTypeSystem.getExpressionProbabilityTable(*context);
  }

  ProbabilityTable<AbstractTypeSystem::StatementKey>
  getStatementProbabilityTable(const Context &context) {
    if (!context)
      return Base::getStatementProbabilityTable(context);
    return subTypeSystem.getStatementProbabilityTable(*context);
  }

private:
  /// Implements the wrapping logic for enabling or disabling the sub type
  /// system.
  template <typename ASTNode>
  static TransferFnArray<ASTNode>
  wrapTransferFns(TransferFnArray<ASTNode> &&array) {
    return mapTuples(
        [](auto &&element) {
          auto unwrapFn =
              [originalTransferFn = std::forward<decltype(element)>(element)](
                  const auto &arg,
                  const TypedContextTuple<ASTNode, Context> &contexts)
              -> Context {
            assert(contexts.back() && "input context is always present");
            // If the input context is enabled, then all transfer functions of
            // the sub type system are enabled.
            if (!contexts.back()->has_value())
              // Otherwise, there is nothing to do except forward an empty
              // optional.
              return std::nullopt;

            return originalTransferFn.template call<SubContext>(
                arg, mapTuplesIntoArray(
                         [](const Context *context) -> const SubContext * {
                           if (!context)
                             return nullptr;

                           assert(context->has_value() &&
                                  "input context being enabled implies the "
                                  "entire sub-tree of contexts being enabled");
                           return &context->value();
                         },
                         contexts));
          };

          using T = std::decay_t<decltype(element)>;
          if constexpr (std::is_same_v<T, OpaqueTransferFn<ASTNode>>) {
            std::vector<std::size_t> indices = element.getInputDependencies();
            return OpaqueTransferFn<ASTNode>(llvm::identity<Context>{},
                                             std::move(indices),
                                             std::move(unwrapFn));
          } else {
            return OpaqueOutputTransferFn<ASTNode>(llvm::identity<Context>{},
                                                   std::move(unwrapFn));
          }
        },
        std::move(array));
  }

  SubTypeSystem subTypeSystem;
};
} // namespace dynamatic::gen
#endif
