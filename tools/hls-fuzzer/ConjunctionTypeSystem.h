#ifndef HLS_FUZZER_CONJUNCTION_TYPE_SYSTEM
#define HLS_FUZZER_CONJUNCTION_TYPE_SYSTEM

#include "TypeSystem.h"
#include <tuple>

#include "Utils.h"
#include "llvm/ADT/STLExtras.h"

namespace dynamatic::gen {

/// Base class for creating type systems by combining multiple independent type
/// systems.
/// The derived class should be specified as the 'Self' parameter while the
/// subtype systems should be specified as 'SubTypeSystems' parameter.
///
/// The typing context of the type system is a tuple of all contexts of the
/// subtype systems. All methods have a default implementation which
/// calls and combines the methods of all subtype systems.
/// 'discard*' calls discard an AST node if any type system discards the node.
/// 'TransferFn' getters combine all transfer functions of the sub-typesystems
/// into a single transfer function with all dependencies merged.
///
/// It is the responsibility of the caller to ensure that no cyclic dependencies
/// occur in any transfer function due to the conjunction of type systems.
template <typename Self, class... SubTypeSystems>
class ConjunctionTypeSystemBase
    : public TypeSystem<std::tuple<typename SubTypeSystems::Context...>, Self> {
public:
  using Base =
      TypeSystem<std::tuple<typename SubTypeSystems::Context...>, Self>;
  using Context = typename Base::Context;

  /// Constructs a conjunctive typesystem from the instances of the
  /// sub-typesystems.
  explicit ConjunctionTypeSystemBase(SubTypeSystems &&...subTypeSystems)
      : typeSystems(std::move(subTypeSystems)...) {}

  TransferFnArray<ast::Function> getFunctionTransferFns() override {
    return combineGetTransferFns<ast::Function>(
        [&](auto &&typeSystem) { return typeSystem.getFunctionTransferFns(); });
  }

  TransferFnArray<ast::ReturnStatement>
  getReturnStatementTransferFns() override {
    return combineGetTransferFns<ast::ReturnStatement>([&](auto &&typeSystem) {
      return typeSystem.getReturnStatementTransferFns();
    });
  }

  bool discardScalarType(const ast::ScalarType &scalarType,
                         const Context &context) {
    return combineDiscard(
        [&](auto &&typeSystem, auto &&context) {
          return typeSystem.discardScalarType(scalarType, context);
        },
        context);
  }

  TransferFnArray<ast::ScalarType> getScalarTypeTransferFns() override {
    return combineGetTransferFns<ast::ScalarType>([&](auto &&typeSystem) {
      return typeSystem.getScalarTypeTransferFns();
    });
  }

  bool discardReturnType(const ast::ReturnType &returnType,
                         const Context &context) {
    return combineDiscard(
        [&](auto &&typeSystem, auto &&context) {
          return typeSystem.discardReturnType(returnType, context);
        },
        context);
  }

  TransferFnArray<ast::ReturnType> getReturnTypeTransferFns() override {
    return combineGetTransferFns<ast::ReturnType>([&](auto &&typeSystem) {
      return typeSystem.getReturnTypeTransferFns();
    });
  }

  bool discardBinaryExpression(ast::BinaryExpression::Op op,
                               const Context &context) {
    return combineDiscard(
        [&](auto &&typeSystem, auto &&context) {
          return typeSystem.discardBinaryExpression(op, context);
        },
        context);
  }

  TransferFnArray<ast::BinaryExpression>
  getBinaryExpressionTransferFns(ast::BinaryExpression::Op op) override {
    return combineGetTransferFns<ast::BinaryExpression>([&](auto &&typeSystem) {
      return typeSystem.getBinaryExpressionTransferFns(op);
    });
  }

  bool discardUnaryExpression(ast::UnaryExpression::Op op,
                              const Context &context) {
    return combineDiscard(
        [&](auto &&typeSystem, auto &&context) {
          return typeSystem.discardUnaryExpression(op, context);
        },
        context);
  }

  TransferFnArray<ast::UnaryExpression>
  getUnaryExpressionTransferFns(ast::UnaryExpression::Op op) override {
    return combineGetTransferFns<ast::UnaryExpression>([&](auto &&typeSystem) {
      return typeSystem.getUnaryExpressionTransferFns(op);
    });
  }

  bool discardVariable(const Context &context) {
    return combineDiscard(
        [&](auto &&typeSystem, auto &&context) {
          return typeSystem.discardVariable(context);
        },
        context);
  }

  TransferFnArray<ast::Variable> getVariableTransferFns() override {
    return combineGetTransferFns<ast::Variable>(
        [&](auto &&typeSystem) { return typeSystem.getVariableTransferFns(); });
  }

  bool discardCastExpression(const Context &context) {
    return combineDiscard(
        [&](auto &&typeSystem, auto &&context) {
          return typeSystem.discardCastExpression(context);
        },
        context);
  }

  TransferFnArray<ast::CastExpression> getCastExpressionTransferFns() override {
    return combineGetTransferFns<ast::CastExpression>([&](auto &&typeSystem) {
      return typeSystem.getCastExpressionTransferFns();
    });
  }

  bool discardConditionalExpression(const Context &context) {
    return combineDiscard(
        [&](auto &&typeSystem, auto &&context) {
          return typeSystem.discardConditionalExpression(context);
        },
        context);
  }

  TransferFnArray<ast::ConditionalExpression>
  getConditionalExpressionTransferFns() override {
    return combineGetTransferFns<ast::ConditionalExpression>(
        [&](auto &&typeSystem) {
          return typeSystem.getConditionalExpressionTransferFns();
        });
  }

  std::optional<ast::Constant> discardConstant(const ast::Constant &constant,
                                               const Context &context) {
    return combineDiscardOptional(
        constant,
        [&](const ast::Constant &constant, auto &&typeSystem, auto &&context) {
          return typeSystem.discardConstant(constant, context);
        },
        context);
  }

  TransferFnArray<ast::Constant> getConstantTransferFns() override {
    return combineGetTransferFns<ast::Constant>(
        [&](auto &&typeSystem) { return typeSystem.getConstantTransferFns(); });
  }

  bool discardExistingScalarParameter(const ast::ScalarParameter &parameter,
                                      const Context &context) {
    return combineDiscard(
        [&](auto &&typeSystem, auto &&context) {
          return typeSystem.discardExistingScalarParameter(parameter, context);
        },
        context);
  }

  TransferFnArray<ast::ExistingScalarParameter>
  getExistingScalarParameterTransferFns() override {
    return combineGetTransferFns<ast::ExistingScalarParameter>(
        [&](auto &&typeSystem) {
          return typeSystem.getExistingScalarParameterTransferFns();
        });
  }

  bool discardFreshScalarParameter(const Context &context) {
    return combineDiscard(
        [&](auto &&typeSystem, auto &&context) {
          return typeSystem.discardFreshScalarParameter(context);
        },
        context);
  }

  TransferFnArray<ast::ScalarParameter>
  getFreshScalarParameterTransferFns() override {
    return combineGetTransferFns<ast::ScalarParameter>([&](auto &&typeSystem) {
      return typeSystem.getFreshScalarParameterTransferFns();
    });
  }

  bool discardArrayReadExpression(const Context &context) {
    return combineDiscard(
        [&](auto &&typeSystem, auto &&context) {
          return typeSystem.discardArrayReadExpression(context);
        },
        context);
  }

  TransferFnArray<ast::ArrayReadExpression>
  getArrayReadExpressionTransferFns() override {
    return combineGetTransferFns<ast::ArrayReadExpression>(
        [&](auto &&typeSystem) {
          return typeSystem.getArrayReadExpressionTransferFns();
        });
  }

  bool discardExistingArrayParameter(const ast::ArrayParameter &parameter,
                                     const Context &context) {
    return combineDiscard(
        [&](auto &&typeSystem, auto &&context) {
          return typeSystem.discardExistingArrayParameter(parameter, context);
        },
        context);
  }

  TransferFnArray<ast::ExistingArrayParameter>
  getExistingArrayParameterTransferFns() override {
    return combineGetTransferFns<ast::ExistingArrayParameter>(
        [&](auto &&typeSystem) {
          return typeSystem.getExistingArrayParameterTransferFns();
        });
  }

  bool discardFreshArrayParameter(const Context &context) {
    return combineDiscard(
        [&](auto &&typeSystem, auto &&context) {
          return typeSystem.discardFreshArrayParameter(context);
        },
        context);
  }

  TransferFnArray<ast::ArrayParameter>
  getFreshArrayParameterTransferFns() override {
    return combineGetTransferFns<ast::ArrayParameter>([&](auto &&typeSystem) {
      return typeSystem.getFreshArrayParameterTransferFns();
    });
  }

  TransferFnArray<ast::ArrayAssignmentStatement>
  getArrayAssignmentStatementTransferFns() override {
    return combineGetTransferFns<ast::ArrayAssignmentStatement>(
        [&](auto &&typeSystem) {
          return typeSystem.getArrayAssignmentStatementTransferFns();
        });
  }

  bool discardStatementList(const Context &context) {
    return combineDiscard(
        [&](auto &&typeSystem, auto &&context) {
          return typeSystem.discardStatementList(context);
        },
        context);
  }

  TransferFnArray<ast::StatementList> getStatementListTransferFns() override {
    return combineGetTransferFns<ast::StatementList>([&](auto &&typeSystem) {
      return typeSystem.getStatementListTransferFns();
    });
  }

  bool discardStructuredForStatement(const Context &context) {
    return combineDiscard(
        [&](auto &&typeSystem, auto &&context) {
          return typeSystem.discardStructuredForStatement(context);
        },
        context);
  }

  TransferFnArray<ast::StructuredForStatement>
  getStructuredForStatementTransferFns() override {
    return combineGetTransferFns<ast::StructuredForStatement>(
        [&](auto &&typeSystem) {
          return typeSystem.getStructuredForStatementTransferFns();
        });
  }

  ProbabilityTable<AbstractTypeSystem::ExpressionKey>
  getExpressionProbabilityTable(const Context &context) {
    return combineExpressionProbabilityTable(
        [](auto &&typeSystem, auto &&context) {
          return typeSystem.getExpressionProbabilityTable(context);
        },
        context);
  }

  ProbabilityTable<AbstractTypeSystem::StatementKey>
  getStatementProbabilityTable(const Context &context) {
    return combineExpressionProbabilityTable(
        [](auto &&typeSystem, auto &&context) {
          return typeSystem.getStatementProbabilityTable(context);
        },
        context);
  }

private:
  /// Calls 'discardCallback' for every typesystem and corresponding context.
  /// Returns true if any of the typesystems discarded the AST node.
  template <typename F>
  bool combineDiscard(F &&discardCallback, const Context &context) {
    return llvm::is_contained(
        mapTuplesIntoArray(discardCallback, typeSystems, context), true);
  }

  /// Variant of 'combineDiscard' for discard methods that consume and return
  /// an ASTNode (e.g. 'ast::Constant').
  /// The initial ASTNode is passed as 'node' and the discard callback is
  /// called once by every type system as long as no empty optional has been
  /// returned.
  template <typename ASTNode, typename F>
  std::optional<ASTNode> combineDiscardOptional(const ASTNode &node,
                                                F &&discardCallback,
                                                const Context &context) {
    std::optional<ASTNode> current = node;
    foreachInTuples(
        [&](auto &&typeSystem, auto &&context) {
          if (!current)
            return;
          current = discardCallback(*current, typeSystem, context);
        },
        typeSystems, context);
    return current;
  }

  /// Calls 'probabilityTableCallback' on every type system and corresponding
  /// context and returns a new probability table that is the combination of
  /// all returned probability tables.
  template <typename F>
  auto combineExpressionProbabilityTable(F &&probabilityTableCallback,
                                         const Context &context) {
    std::array probabilityTables =
        mapTuplesIntoArray(probabilityTableCallback, typeSystems, context);
    for (auto &iter : llvm::drop_begin(probabilityTables))
      probabilityTables[0].inplaceMerge(iter);

    return std::move(probabilityTables[0]);
  }

  /// Calls 'transferFnsCallback' on every typesystem and combines all
  /// 'TransferFnArray<ASTNode>' returned into one single
  /// 'TransferFnArray<ASTNode>'.
  template <typename ASTNode, typename F>
  TransferFnArray<ASTNode> combineGetTransferFns(F &&transferFnsCallback) {
    // Tuple of 'TransferFnArray<ASTNode>'s, one for each typeSystem.
    auto transferFnsPerTypeSystem = mapTuples(transferFnsCallback, typeSystems);
    auto outputTransferFns = mapTuplesIntoArray(
        [](auto &&dep) {
          return std::move(std::get<OutputTransferFn<ASTNode>>(dep));
        },
        transferFnsPerTypeSystem);

    // Construct the transfer array by constructing an 'OpaqueTransferFn' for
    // every subelement.
    return mapTuplesInto(
        [&](auto &&...args) {
          return TransferFnArray<ASTNode>{
              /*subelements=*/std::forward<decltype(args)>(args)...,
              /*output=*/
              combineOutputTransferFns(std::move(outputTransferFns))};
        },
        [&](auto index) {
          constexpr std::size_t indexWithinTransferFnArray = decltype(index){};

          auto transferFnsForSubElement = mapTuplesIntoArray(
              [&](auto &&dep) -> OpaqueTransferFn<ASTNode> {
                return std::move(std::get<indexWithinTransferFnArray>(dep));
              },
              transferFnsPerTypeSystem);

          return combineOpaqueTransferFns(std::move(transferFnsForSubElement));
        },
        getTupleOfIndices(std::make_index_sequence<
                          std::tuple_size_v<typename ASTNode::SubElements>>{}));
  }

  /// Combines all 'OpaqueTransferFn' in 'transferFnPerTypeSystem' into a single
  /// 'OpaqueTransferFn'.
  template <typename ASTNode>
  static OpaqueTransferFn<ASTNode> combineOpaqueTransferFns(
      std::array<OpaqueTransferFn<ASTNode>, sizeof...(SubTypeSystems)>
          &&transferFnPerTypeSystem) {
    std::vector<std::size_t> indices;
    foreachInTuples(
        [&](auto &&transferFn) {
          llvm::append_range(indices, transferFn.getInputDependencies());
        },
        transferFnPerTypeSystem);
    // Remove duplicates.
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());

    using DepT = std::decay_t<decltype(transferFnPerTypeSystem)>;
    return OpaqueTransferFn<ASTNode>(
        std::move(indices), std::move(transferFnPerTypeSystem),
        +[](const std::any &any,
            const typename OpaqueTransferFn<ASTNode>::SubElementsTuple
                &subElements,
            const typename OpaqueTransferFn<ASTNode>::ContextTuple &contexts)
            -> OpaqueContext {
          const auto &transferFns = std::any_cast<DepT>(any);

          // Call all sub-transfer functions with their respective contexts.
          return OpaqueContext(enumerateTuples(
              [&](auto &&typeSystemIndexT, auto &&transferFn) {
                constexpr std::size_t typeSystemIndex =
                    decltype(typeSystemIndexT){};

                auto subContexts =
                    contextTupleToSubContextTuple<ASTNode, typeSystemIndex>(
                        contexts);

                return transferFn(subElements, subContexts)
                    .template cast<
                        std::tuple_element_t<typeSystemIndex, Context>>();
              },
              transferFns));
        });
  }

  /// Combines all 'OutputTransferFn' in 'outputTransferFnPerTypeSystem' into a
  /// single 'OutputTransferFn'.
  template <typename ASTNode>
  static OutputTransferFn<ASTNode> combineOutputTransferFns(
      std::array<OutputTransferFn<ASTNode>, sizeof...(SubTypeSystems)>
          &&outputTransferFnPerTypeSystem) {
    return OutputTransferFn<ASTNode>(
        [outputTransferFnPerTypeSystem =
             std::move(outputTransferFnPerTypeSystem)](
            const ASTNode &astNode,
            const typename OpaqueTransferFn<ASTNode>::ContextTuple
                &contextTuple) -> OpaqueContext {
          return OpaqueContext(enumerateTuples(
              [&](auto typeSystemIndexT, auto &&transferFn) {
                constexpr std::size_t typeSystemIndex =
                    decltype(typeSystemIndexT){};

                auto subContexts =
                    contextTupleToSubContextTuple<ASTNode, typeSystemIndex>(
                        contextTuple);

                return transferFn(astNode, subContexts)
                    .template cast<
                        std::tuple_element_t<typeSystemIndex, Context>>();
              },
              outputTransferFnPerTypeSystem));
        });
  }

  template <typename ASTNode, std::size_t index>
  static auto contextTupleToSubContextTuple(
      const typename OpaqueTransferFn<ASTNode>::ContextTuple &contexts) {
    return mapTuplesIntoArray(
        [&](auto &&context) -> std::optional<OpaqueContext> {
          if (!context)
            return std::nullopt;

          return OpaqueContext(
              std::get<index>(context->template cast<Context>()));
        },
        contexts);
  }

  std::tuple<SubTypeSystems...> typeSystems;
};

/// Non-abstract version of 'ConjunctionTypeSystemBase' that can be instantiated
/// directly without deriving from it.
template <class... SubTypeSystems>
class ConjunctionTypeSystem final
    : public ConjunctionTypeSystemBase<ConjunctionTypeSystem<SubTypeSystems...>,
                                       SubTypeSystems...> {
public:
  using ConjunctionTypeSystemBase<ConjunctionTypeSystem,
                                  SubTypeSystems...>::ConjunctionTypeSystemBase;
};

} // namespace dynamatic::gen

#endif
