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

  ConjunctionTypeSystemBase() = default;

  /// Constructs a conjunctive typesystem from the instances of the
  /// sub-typesystems.
  explicit ConjunctionTypeSystemBase(SubTypeSystems &&...subTypeSystems)
      : typeSystems(std::move(subTypeSystems)...) {}

  TransferFnArray<ast::Function> getFunctionTransferFns() override {
    return combineGetTransferFns(
        [&](auto &&typeSystem) { return typeSystem.getFunctionTransferFns(); });
  }

  TransferFnArray<ast::ReturnStatement>
  getReturnStatementTransferFns() override {
    return combineGetTransferFns([&](auto &&typeSystem) {
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
    return combineGetTransferFns([&](auto &&typeSystem) {
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
    return combineGetTransferFns([&](auto &&typeSystem) {
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
    return combineGetTransferFns([&](auto &&typeSystem) {
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
    return combineGetTransferFns([&](auto &&typeSystem) {
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
    return combineGetTransferFns(
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
    return combineGetTransferFns([&](auto &&typeSystem) {
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
    return combineGetTransferFns([&](auto &&typeSystem) {
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
    return combineGetTransferFns(
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
    return combineGetTransferFns([&](auto &&typeSystem) {
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
    return combineGetTransferFns([&](auto &&typeSystem) {
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
    return combineGetTransferFns([&](auto &&typeSystem) {
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
    return combineGetTransferFns([&](auto &&typeSystem) {
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

  std::optional<std::size_t> discardArrayDimension(std::size_t dimension,
                                                   const Context &context) {
    return combineDiscardOptional(
        dimension,
        [&](std::size_t dimension, auto &&typeSystem, auto &&context) {
          return typeSystem.discardArrayDimension(dimension, context);
        },
        context);
  }

  TransferFnArray<ast::ArrayParameter>
  getFreshArrayParameterTransferFns() override {
    return combineGetTransferFns([&](auto &&typeSystem) {
      return typeSystem.getFreshArrayParameterTransferFns();
    });
  }

  TransferFnArray<ast::ArrayAssignmentStatement>
  getArrayAssignmentStatementTransferFns() override {
    return combineGetTransferFns([&](auto &&typeSystem) {
      return typeSystem.getArrayAssignmentStatementTransferFns();
    });
  }

  bool discardScalarAssignmentStatement(const Context &context) {
    return combineDiscard(
        [&](auto &&typeSystem, auto &&context) {
          return typeSystem.discardScalarAssignmentStatement(context);
        },
        context);
  }

  TransferFnArray<ast::ScalarAssignmentStatement>
  getScalarAssignmentStatementTransferFns() override {
    return combineGetTransferFns([&](auto &&typeSystem) {
      return typeSystem.getScalarAssignmentStatementTransferFns();
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
    return combineGetTransferFns([&](auto &&typeSystem) {
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
    return combineGetTransferFns([&](auto &&typeSystem) {
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

protected:
  /// Class used to denote dependencies on both other sub elements and a
  /// specific type system of theirs.
  template <typename TypeSystem, std::size_t i>
  struct Dep {
    using type = TypeSystem;
    constexpr static std::size_t index = i;
  };

  /// Utility function that can be used to cross transfer function dependencies
  /// between different type systems.
  /// By default, every type system is run independently from each other with
  /// contexts of each having no influence on any others.
  ///
  /// This method makes it possible for the class implementing the conjunction
  /// to implement logic that modifies the input context of 'subElement'
  /// of an 'ASTNode' (deduced from its transfer functions).
  ///
  /// Use cases e.g. include being able to enable an optional type system for
  /// 'subElement' based on a context value of another type system or having one
  /// type system that performs analysis, another that heavily restricts the
  /// generated program and letting the analysis inform the latter how to now
  /// restrict the generation of 'subElement'.
  ///
  /// 'f' is a callable of the form:
  ///   SubTypeSystem::Context(const DepTypeSystemContext&...).
  /// where 'SubTypeSystem' is the type system whose context should be modified.
  /// The arguments are deduced from 'dependencies' which must be instances
  /// of 'Dep'.
  /// It allows adding extra dependencies on other 'ASTNode's context of a
  /// specific type system.
  ///
  /// Example:
  ///   crossTransferFns<ast::ScalarAssignment::VALUE, BitWidthTypeSystem,
  ///                    Dep<LoopRecognitionTypeSystem, INPUT_DEPENDENCY>>
  ///                    (scalarAssignmentTransferFns,
  ///                     [](const LoopRecognitionContext& context) {
  ///                       if (context.isInnerMostLoop)
  ///                         return BitWidthContext{.bitwidth = 5};
  ///                       return BitWidthContext{.bitwidth = 10};
  ///                    });
  ///
  /// This transfer function crossing e.g. makes it such that the bitwidth of
  /// expressions within values of a scalar assignment is 5 iff in an innermost
  /// loop (as determined by the 'LoopRecognitionTypeSystem'), 10 otherwise.
  ///
  /// Unlike in e.g. 'TransferFn's, the dependencies:
  /// * Do not capture the AST elements (this should be done by the sub type
  ///   systems)
  /// * May depend on themselves (e.g. the example above is allowed to have a
  ///   Dep<..., ast::ScalarAssignment::VALUE> without causing a cycle).
  ///   In that case the input context that the given type system calculated
  ///   for 'subElement' is returned.
  ///
  /// It's the users responsibility to make sure no cyclic dependencies are
  /// introduced and that the contexts are modified in a way that holds up the
  /// given type systems invariants.
  template <std::size_t subElement, typename SubTypeSystem,
            typename... dependencies, typename TransferFns, typename F>
  static void crossTransferFns(TransferFns &transferFnArray, F &&f) {
    using ASTNode = TransferFnArrayASTNode<TransferFns>;
    static_assert(subElement < std::tuple_size_v<typename ASTNode::SubElements>,
                  "'subElement' must refer to a subelement of 'ASTNode'");

    // Self-references in dependencies are not legal in 'wrap' as the resulting
    // node would depend on itself and cause a cycle!
    // For that reason we filter them out here and reinsert them later.
    using CalcFilter = FilterDeps<subElement, dependencies...>;
    // Filtered 'Dep' instances without 'subElement'.
    using Tuple = typename CalcFilter::value;

    auto &transferFn = std::get<subElement>(transferFnArray);

    std::apply(
        [&](auto &&...newDeps) {
          transferFn =
              std::move(transferFn)
                  .template wrap<Context,
                                 std::decay_t<decltype(newDeps)>::index...>(
                      [f = std::forward<F>(f)](
                          llvm::function_ref<Context()> wrapped,
                          const auto &...args) {
                        // Previous context.
                        Context context = wrapped();

                        // First remove the 'ASTNode's from the argument list.
                        // This is now equal to just the context's without
                        // the self references.
                        auto contextsOnly = mapTuplesInto(
                            [&](auto &&...args) {
                              // Create one flat tuple out of the tuples.
                              return std::tuple_cat(
                                  std::forward<decltype(args)>(args)...);
                            },
                            [&](auto &&arg) {
                              using T = std::decay_t<decltype(arg)>;
                              if constexpr (std::is_same_v<Context, T>) {

                                return std::forward_as_tuple(arg);
                              } else {
                                return std::make_tuple();
                              }
                            },
                            std::forward_as_tuple(args...));

                        struct Sentinel {};

                        // Now reinsert 'context' at the corresponding indices
                        // of where the original dep was prior to it being
                        // filtered.
                        // Note that we pad the context tuple to contain some
                        // extra sentinel values such that its length is equal
                        // to 'depdencies' again, creating a one to one
                        // correspondence.
                        auto withSelfRefs = enumerateTuplesInto(
                            [&](auto &&...args) {
                              // Create one flat tuple out of the tuples.
                              return std::tuple_cat(
                                  std::forward<decltype(args)>(args)...);
                            },
                            [&](auto &&indexT, auto &&arg) {
                              constexpr std::size_t index =
                                  std::decay_t<decltype(indexT)>{};
                              using T = std::decay_t<decltype(arg)>;
                              if constexpr (CalcFilter::matches(index)) {
                                if constexpr (std::is_same_v<Sentinel, T>)
                                  return std::forward_as_tuple(context);
                                else
                                  return std::forward_as_tuple(context, arg);
                              } else {
                                return std::forward_as_tuple(arg);
                              }
                            },
                            std::tuple_cat(
                                std::move(contextsOnly),
                                // Padding.
                                repeatInTuple<(
                                    sizeof...(dependencies) -
                                    std::tuple_size_v<decltype(contextsOnly)>)>(
                                    Sentinel{})));

                        std::get<typename SubTypeSystem::Context>(context) =
                            std::apply(
                                f,
                                // Finally, return the context of just the
                                // requested type systems.
                                mapTuplesInto(
                                    [](auto &&...args) {
                                      return std::forward_as_tuple(
                                          std::forward<decltype(args)>(
                                              args)...);
                                    },
                                    [](const Context &context,
                                       auto &&dep) -> decltype(auto) {
                                      using Dep = std::decay_t<decltype(dep)>;
                                      return std::get<
                                          typename Dep::type::Context>(context);
                                    },
                                    withSelfRefs,
                                    std::make_tuple(dependencies{}...)));
                        return context;
                      });
        },
        Tuple{});
  }

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
  template <typename F>
  auto combineGetTransferFns(F &&transferFnsCallback) {
    // Tuple of 'TransferFnArray<ASTNode>'s, one for each typeSystem.
    auto transferFnsPerTypeSystem = mapTuples(transferFnsCallback, typeSystems);
    using ASTNode = TransferFnArrayASTNode<
        std::tuple_element_t<0, decltype(transferFnsPerTypeSystem)>>;
    auto outputTransferFns = mapTuplesIntoArray(
        [](auto &&dep) {
          return std::move(std::get<OpaqueOutputTransferFn<ASTNode>>(dep));
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

private:
  /// Filters out instances of 'Dep<..., subElement>'.
  /// Returns the result as a tuple of the remaining 'Dep' instances within
  /// 'value'.
  /// The constexpr function 'matches' returns true iff the given index used
  /// to be a 'Dep<..., subElement>'.
  template <std::size_t subElement, typename... dependencies>
  struct FilterDeps;

  // End of recursion.
  template <std::size_t subElement>
  struct FilterDeps<subElement> {
    using value = std::tuple<>;

    constexpr static bool matches(std::size_t) { return false; }
  };

  // Front of the list is a 'Dep<..., subElement>'.
  template <std::size_t subElement, typename TypeSystem,
            typename... dependencies>
  struct FilterDeps<subElement, Dep<TypeSystem, subElement>, dependencies...> {
    // Get the value from the rest of the list, deliberately skipping
    // 'Dep<..., subElement>'.
    using value = typename FilterDeps<subElement, dependencies...>::value;

    constexpr static bool matches(std::size_t index) {
      // If index is 0 at this point its a match.
      return index == 0 ||
             // Otherwise we recurse into the sublist.
             FilterDeps<subElement, dependencies...>::matches(index - 1);
    }
  };

  // No instance found default case.
  template <std::size_t subElement, typename Front, typename... dependencies>
  struct FilterDeps<subElement, Front, dependencies...> {
    // Prepend 'Front'.
    using value = decltype(std::tuple_cat(
        std::declval<std::tuple<Front>>(),
        std::declval<
            typename FilterDeps<subElement, dependencies...>::value>()));

    // Check whether index matches the sub list.
    constexpr static bool matches(std::size_t index) {
      return FilterDeps<subElement, dependencies...>::matches(index - 1);
    }
  };

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

    return OpaqueTransferFn<ASTNode>(
        llvm::identity<Context>{}, std::move(indices),
        [transferFnPerTypeSystem = std::move(transferFnPerTypeSystem)](
            const SubElementsTuple<ASTNode> &subElements,
            const TypedContextTuple<ASTNode, Context> &contexts) -> Context {
          // Call all sub-transfer functions with their respective contexts.
          return enumerateTuples(
              [&](auto &&typeSystemIndexT,
                  const OpaqueTransferFn<ASTNode> &transferFn) {
                constexpr std::size_t typeSystemIndex =
                    decltype(typeSystemIndexT){};

                auto subContexts =
                    contextTupleToSubContextTuple<ASTNode, typeSystemIndex>(
                        contexts);

                return transferFn.template call<
                    std::tuple_element_t<typeSystemIndex, Context>>(
                    subElements, subContexts);
              },
              transferFnPerTypeSystem);
        });
  }

  /// Combines all 'OutputTransferFn' in 'outputTransferFnPerTypeSystem' into a
  /// single 'OutputTransferFn'.
  template <typename ASTNode>
  static OpaqueOutputTransferFn<ASTNode> combineOutputTransferFns(
      std::array<OpaqueOutputTransferFn<ASTNode>, sizeof...(SubTypeSystems)>
          &&outputTransferFnPerTypeSystem) {
    return OpaqueOutputTransferFn<ASTNode>(
        llvm::identity<Context>{},
        [outputTransferFnPerTypeSystem =
             std::move(outputTransferFnPerTypeSystem)](
            const ASTNode &astNode,
            const TypedContextTuple<ASTNode, Context> &contextTuple) {
          return enumerateTuples(
              [&](auto typeSystemIndexT,
                  const OpaqueOutputTransferFn<ASTNode> &transferFn) {
                constexpr std::size_t typeSystemIndex =
                    decltype(typeSystemIndexT){};

                auto subContexts =
                    contextTupleToSubContextTuple<ASTNode, typeSystemIndex>(
                        contextTuple);

                return transferFn.template call<
                    std::tuple_element_t<typeSystemIndex, Context>>(
                    astNode, subContexts);
              },
              outputTransferFnPerTypeSystem);
        });
  }

  template <typename ASTNode, std::size_t index>
  static auto contextTupleToSubContextTuple(
      const TypedContextTuple<ASTNode, Context> &contexts) {
    return mapTuplesIntoArray(
        [&](const Context *context)
            -> const std::tuple_element_t<index, Context> * {
          if (!context)
            return nullptr;

          return &std::get<index>(*context);
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
