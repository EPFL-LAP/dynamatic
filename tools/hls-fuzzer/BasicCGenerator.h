#ifndef DYNAMATIC_HLS_FUZZER_BasicCProducer
#define DYNAMATIC_HLS_FUZZER_BasicCProducer

#include "AST.h"
#include "Randomly.h"
#include "TypeSystem.h"
#include "Utils.h"

#include "llvm/ADT/ScopeExit.h"

#include <optional>

namespace dynamatic::gen {

/// Base generator which is responsible for generating valid C programs that do
/// not contain undefined behavior.
///
/// The generated programs can be further restricted through the use of a type
/// system which constrains the kind of valid expressions the generator may
/// output.
///
/// Generating new C constructs should be implemented in this class.
class BasicCGenerator {
public:
  /// Constructs a new base generator which generates programs that adhere to
  /// the given type system.
  /// 'entryContext' is entry state used to type check the function returned by
  /// 'generate'.
  template <class TypingContext, class Self>
  explicit BasicCGenerator(Randomly &random,
                           TypeSystem<TypingContext, Self> &typeSystem,
                           const TypingContext &entryContext = {})
      : random(random), typeSystem(typeSystem), entryContext(entryContext) {
    initGenerators();
  }

  /// Generates an entire C program that can compile and run.
  void generate(llvm::raw_ostream &os, std::string_view functionName);

  /// Returns a new function with the given function name.
  ast::Function generateFunction(llvm::StringRef functionName);

  /// Generates a dynamatic test bench for the given function.
  // TODO: Could return a function once our AST is powerful enough to represent
  // entire test benches.
  std::string generateTestBench(const ast::Function &kernel) const;

private:
  std::string generateFreshVarName() {
    return "var" + std::to_string(varCounter++);
  }

  std::pair<ast::ReturnStatement, OpaqueContext>
  generateReturnStatement(const OpaqueContext &constraints);

  std::pair<ast::Expression, OpaqueContext>
  generateExpression(const OpaqueContext &context);

  std::optional<std::pair<ast::Expression, OpaqueContext>>
  generateBinaryExpression(ast::BinaryExpression::Op op,
                           const OpaqueContext &constraints);

  std::optional<std::pair<ast::Expression, OpaqueContext>>
  generateUnaryExpression(ast::UnaryExpression::Op op,
                          const OpaqueContext &context);

  std::optional<std::pair<ast::ConditionalExpression, OpaqueContext>>
  generateConditionalExpression(const OpaqueContext &constraint);

  std::optional<std::pair<ast::CastExpression, OpaqueContext>>
  generateCastExpression(const OpaqueContext &constraint);

  ast::Constant getConstantForType(const ast::ScalarType &scalarType) const;

  std::optional<std::pair<ast::Constant, OpaqueContext>>
  generateConstant(const OpaqueContext &constraint) const;

  std::optional<std::pair<ast::ArrayReadExpression, OpaqueContext>>
  generateArrayReadExpression(const OpaqueContext &context);

  std::optional<std::pair<ast::ArrayParameter, OpaqueContext>>
  generateArrayParameter(const OpaqueContext &context);

  std::optional<std::pair<ast::Variable, OpaqueContext>>
  generateScalarParameter(const OpaqueContext &constraints);

  /// Generates a scalar type or none if it was impossible to generate a scalar
  /// type in the given context.
  /// 'toExclude' may be supplied by the caller to further exclude some scalar
  /// types based on the given context.
  std::optional<std::pair<ast::ScalarType, OpaqueContext>> generateScalarType(
      const OpaqueContext &context,
      llvm::function_ref<bool(const ast::ScalarType &)> toExclude =
          nullptr) const;

  std::pair<ast::ReturnType, OpaqueContext>
  generateReturnType(const OpaqueContext &context) const;

  std::pair<ast::StatementList, OpaqueContext>
  generateStatementList(const OpaqueContext &context);

  std::optional<std::pair<ast::Statement, OpaqueContext>>
  generateStatement(const OpaqueContext &context);

  std::optional<std::pair<ast::ArrayAssignmentStatement, OpaqueContext>>
  generateArrayAssignmentStatement(const OpaqueContext &context);

  std::optional<std::pair<ast::StructuredForStatement, OpaqueContext>>
  generateStructuredForStatement(const OpaqueContext &context);

  Randomly &random;
  std::optional<ast::ReturnType> maybeReturnType;
  std::vector<ast::ScalarParameter> scalarParameters;
  std::vector<std::vector<ast::ScalarParameter>> localVariableStack;
  std::vector<ast::ArrayParameter> arrayParameters;
  std::size_t varCounter = 0;
  AbstractTypeSystem &typeSystem;
  OpaqueContext entryContext;

  [[nodiscard]] auto pushNewScope() {
    localVariableStack.emplace_back();
    return llvm::make_scope_exit([&] { localVariableStack.pop_back(); });
  }

  void addVariable(ast::ScalarType type, std::string name) {
    localVariableStack.back().emplace_back(std::move(type), std::move(name));
  }

  template <typename ASTNode>
  using Constructor =
      std::function<std::optional<std::pair<ASTNode, OpaqueContext>>(
          BasicCGenerator *, const OpaqueContext &)>;

  template <typename ASTNode, typename Key>
  using ConstructorKeyPair = std::pair<Constructor<ASTNode>, Key>;

  std::vector<
      ConstructorKeyPair<ast::Expression, AbstractTypeSystem::ExpressionKey>>
      expressionGenerators;
  std::vector<
      ConstructorKeyPair<ast::Statement, AbstractTypeSystem::StatementKey>>
      statementGenerators;

  void initGenerators();

  /// Returns a tuple of 'std::integral_constant's for every element in 'is'.
  template <std::size_t... is>
  constexpr static auto getIndicesTuple(std::index_sequence<is...>) {
    return std::tuple{std::integral_constant<std::size_t, is>{}...};
  }

  template <typename ASTNode, typename = typename ASTNode::SubElements>
  struct GenerateWithDependencies;

  template <typename ASTNode, typename... SubElements>
  struct GenerateWithDependencies<ASTNode, std::tuple<SubElements...>> {

    /// Convenience overload for terminal ASTNodes. Calculates the output
    /// context and returns the node 'node' as is.
    template <std::size_t size = sizeof...(SubElements),
              std::enable_if_t<size == 0> * = nullptr>
    std::pair<ASTNode, OpaqueContext>
    operator()(const OpaqueContext &inputContext,
               const TransferFnArray<ASTNode> &transferFunctions,
               ASTNode node) const {
      return std::move(*(*this)(inputContext, transferFunctions,
                                [&] { return std::move(node); }));
    }

    std::optional<std::pair<ASTNode, OpaqueContext>> operator()(
        const OpaqueContext &parentContext,
        const TransferFnArray<ASTNode> &transferFunctions,
        llvm::function_ref<std::optional<std::pair<SubElements, OpaqueContext>>(
            OpaqueContext)>... generators,
        llvm::function_ref<std::optional<ASTNode>(SubElements &&...)>
            constructor) const {
      typename OpaqueTransferFn<ASTNode>::SubElementsTuple subElements;

      typename OpaqueTransferFn<ASTNode>::ContextTuple contexts;
      std::get<sizeof...(SubElements)>(contexts) = parentContext;

      // Calculate a topological order between all dependencies.
      // To do so we use a worklist of elements whose dependencies are all
      // satisfied and an edge list that for every node 'i', contains all
      // outgoing edges.
      // This is opposite from 'OpaqueTransferFn' which returns the incoming
      // edges.

      // Note: We use 'std::array' here everywhere since the bounds are known
      // and small.
      using NodeList = std::array<std::size_t, sizeof...(SubElements)>;
      std::size_t workListSize = 0;
      NodeList worklist{};

      // For a given node 'i', contains the number of outgoing edges from that
      // node.
      NodeList forwardEdgeCount{};
      // For a given node 'i', contains the destinations of each outgoing edge
      // from that node.
      std::array<NodeList, sizeof...(SubElements)> forwardEdgeList{};
      // For a given node 'i', contains the number of incoming edges into 'i'.
      NodeList incomingEdgeCount{};

      std::apply(
          [&](auto &&...elementIndices) {
            (
                [&](auto elementIndex) {
                  constexpr std::size_t index = decltype(elementIndex){};
                  auto &iter = std::get<index>(transferFunctions);

                  if (iter.getInputDependencies().empty() ||
                      iter.getInputDependencies() ==
                          llvm::ArrayRef{INPUT_DEPENDENCY}) {
                    // No dependency (besides the parent context which is
                    // satisfied).
                    worklist[workListSize++] = index;
                    return;
                  }

                  // Build the outgoing edge list but do keep track of the
                  // number of incoming edges.
                  for (auto fromIndex : iter.getInputDependencies())
                    if (fromIndex != INPUT_DEPENDENCY) {
                      forwardEdgeList[fromIndex]
                                     [forwardEdgeCount[fromIndex]++] = index;
                      ++incomingEdgeCount[index];
                    }
                }(elementIndices),
                ...);
          },
          getIndicesTuple(std::index_sequence_for<SubElements...>{}));

      std::size_t topoOrderSize = 0;
      NodeList topoOrder{};
      while (workListSize > 0) {
        std::size_t index = worklist[--workListSize];
        topoOrder[topoOrderSize++] = index;
        // "Remove" all outgoing edges from 'index'.
        // If a node has no more incoming edges, then it can be scheduled and
        // added to the worklist.
        for (auto &&m : llvm::ArrayRef(forwardEdgeList[index])
                            .take_front(forwardEdgeCount[index]))
          if (--incomingEdgeCount[m] == 0)
            worklist[workListSize++] = m;
      }

      assert(topoOrderSize == sizeof...(SubElements) &&
             "transfer function dependency graph contains cycles");

      // Finally, generate the subelements in topological order.
      for (std::size_t iter : topoOrder) {
        // We need to use fold-expressions over compile time constants to be
        // able to index into 'contexts' and 'subElements'.
        // The conditional-expressions are just if-conditions that perform a
        // given assignment if 'iter' matches that current 'index'.
        bool success = std::apply(
            [&](auto &&...indices) {
              return ([&](auto indexT) {
                if (iter != indexT)
                  return true;

                constexpr std::size_t index = decltype(indexT){};

                auto &context = std::get<index>(contexts);
                // First calculate the context for the subelement.
                context.emplace(
                    std::get<indexT>(transferFunctions)(subElements, contexts));

                std::optional result =
                    std::get<index>(std::make_tuple(generators...))(*context);
                if (!result)
                  return false;

                // Update with the output context plus generated subelement.
                std::tie(std::get<index>(subElements), context) =
                    std::move(*result);
                return true;
              }(indices) &&
                      ...);
            },
            getTupleOfIndices(std::index_sequence_for<SubElements...>{}));

        // Discard this AST node if we failed to generate a subelement.
        if (!success)
          return std::nullopt;
      }

      // Call the constructor with all subelements.
      // It should be safe to dereference all optionals since they have been
      // guaranteed to have been generated.
      std::optional<ASTNode> astNode = std::apply(
          [&](auto &&...values) { return constructor(std::move(*values)...); },
          std::move(subElements));
      if (!astNode)
        return std::nullopt;

      // Calculate the output context.
      OpaqueContext outputContext = std::get<OutputTransferFn<ASTNode>>(
          transferFunctions)(*astNode, contexts);
      return std::pair{std::move(*astNode), std::move(outputContext)};
    }
  };

  /// Callable object used to generate an 'ASTNode' from its subelements.
  /// The signature of the object can be thought of as:
  ///
  /// (const OpaqueContext &parentContext,
  ///  const DependencyArray<ASTNode> &dependencies,
  ///  llvm::function_ref<
  ///      std::optional<SubElements>(OpaqueContext)>... generators,
  ///  llvm::function_ref<std::optional<ASTNode>(SubElements &&...)>
  ///      constructor) -> std::optional<ASTNode>
  ///  where 'SubElements' are the subelements of 'ASTNode' specified in
  ///  'TypeSystemTraits<ASTNode>::SubElements'.
  ///
  /// 'parentContext' is the input context, 'generators' are callbacks to
  /// generate every corresponding subelement of 'ASTNode' and 'constructor'
  /// the final callback to construct 'ASTNode' from the subelements.
  template <typename ASTNode>
  constexpr static auto generateWithDependencies =
      GenerateWithDependencies<ASTNode>{};
};

} // namespace dynamatic::gen

#endif
