#ifndef DYNAMATIC_HLS_FUZZER_BasicCProducer
#define DYNAMATIC_HLS_FUZZER_BasicCProducer

#include "AST.h"
#include "Randomly.h"
#include "TypeSystem.h"

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
      : random(random), typeSystem(typeSystem), entryContext(entryContext) {}

  /// Returns a new function with the given function name.
  ast::Function generate(std::string_view functionName);

  /// Generates a dynamatic test bench for the given function.
  // TODO: Could return a function once our AST is powerful enough to represent
  // entire test benches.
  std::string generateTestBench(const ast::Function &kernel) const;

private:
  std::string generateFreshVarName() {
    return "var" + std::to_string(varCounter++);
  }

  friend class PendingParameter;

  /// Convenience class that automatically undoes the creation of a parameter
  /// unless it was committed.
  class PendingParameter {
  public:
    PendingParameter(BasicCGenerator &generator,
                     const ast::ScalarParameter &parameter)
        : generator(generator), parameter(parameter) {}

    ~PendingParameter() {
      if (!parameter)
        return;

      generator.scalarParameters.pop_back();
      generator.varCounter--;
    }

    const ast::ScalarParameter &getParameter() const {
      assert(parameter && "must not yet be committed");
      return *parameter;
    }

    ast::ScalarParameter commit() {
      assert(parameter && "must not yet be committed");
      ast::ScalarParameter value = std::move(*parameter);
      parameter.reset();
      return value;
    }

  private:
    BasicCGenerator &generator;
    std::optional<ast::ScalarParameter> parameter;
  };

  PendingParameter generateFreshScalarParameter(ast::ScalarType datatype,
                                                const OpaqueContext &context);

  ast::ReturnStatement
  generateReturnStatement(const OpaqueContext &constraints);

  ast::Expression generateExpression(const OpaqueContext &context,
                                     std::size_t depth);

  std::optional<ast::Expression>
  generateBinaryExpression(ast::BinaryExpression::Op op,
                           const OpaqueContext &constraints, std::size_t depth);

  std::optional<ast::Expression>
  generateUnaryExpression(ast::UnaryExpression::Op op,
                          const OpaqueContext &context, std::size_t depth);

  std::optional<ast::ConditionalExpression>
  generateConditionalExpression(const OpaqueContext &constraint,
                                std::size_t depth);

  std::optional<ast::CastExpression>
  generateCastExpression(const OpaqueContext &constraint, std::size_t depth);

  std::optional<ast::Constant> generateConstant(const OpaqueContext &constraint,
                                                std::size_t depth = 0) const;

  std::optional<ast::ArrayReadExpression>
  generateArrayReadExpression(const OpaqueContext &context,
                              std::size_t depth = 0);

  std::optional<ast::ArrayParameter>
  generateArrayParameter(const OpaqueContext &context, std::size_t depth = 0);

  std::optional<ast::Variable>
  generateScalarParameter(const OpaqueContext &constraints,
                          std::size_t depth = 0);

  /// Generates a scalar type or none if it was impossible to generate a scalar
  /// type in the given context.
  /// 'toExclude' may be supplied by the caller to further exclude some scalar
  /// types based on the given context.
  std::optional<ast::ScalarType> generateScalarType(
      const OpaqueContext &context,
      llvm::function_ref<bool(const ast::ScalarType &)> toExclude =
          nullptr) const;

  ast::ReturnType generateReturnType(const OpaqueContext &context) const;

  std::vector<ast::Statement>
  generateStatementList(const OpaqueContext &context);

  std::optional<ast::Statement> generateStatement(const OpaqueContext &context);

  std::optional<ast::ArrayAssignmentStatement>
  generateArrayAssignmentStatement(const OpaqueContext &context);

  Randomly &random;
  ast::ReturnType returnType;
  std::vector<std::pair<ast::ScalarParameter, OpaqueContext>> scalarParameters;
  std::vector<std::pair<ast::ArrayParameter, OpaqueContext>> arrayParameters;
  std::size_t varCounter = 0;
  AbstractTypeSystem &typeSystem;
  OpaqueContext entryContext;

  /// Returns a tuple of 'std::integral_constant's for every element in 'is'.
  template <std::size_t... is>
  constexpr static auto getIndicesTuple(std::index_sequence<is...>) {
    return std::tuple{std::integral_constant<std::size_t, is>{}...};
  }

  template <typename ASTNode,
            typename = typename TypeSystemTraits<ASTNode>::SubElements>
  struct GenerateWithDependencies;

  template <typename ASTNode, typename... SubElements>
  struct GenerateWithDependencies<ASTNode, std::tuple<SubElements...>> {
    std::optional<ASTNode>
    operator()(const OpaqueContext &parentContext,
               const DependencyArray<ASTNode> &dependencies,
               llvm::function_ref<
                   std::optional<SubElements>(OpaqueContext)>... generators,
               llvm::function_ref<std::optional<ASTNode>(SubElements &&...)>
                   constructor) const {
      typename OpaqueDependency<ASTNode>::SubElementsTuple subElements;

      // TODO: For now subelement generators cannot yet return an output
      //       context. We assume output context == input context.
      typename OpaqueDependency<ASTNode>::ContextTuple contexts;
      std::get<sizeof...(SubElements)>(contexts) = parentContext;

      // Calculate a topological order between all dependencies.
      // To do so we use a worklist of elements whose dependencies are all
      // satisfied and an edge list that for every node 'i', contains all
      // outgoing edges.
      // This is opposite from 'OpaqueDependency' which returns the incoming
      // edges.

      // Note: We use 'std::array' here everywhere since the bounds are known
      // and small.
      std::size_t workListSize = 0;
      std::array<std::size_t, sizeof...(SubElements)> worklist;

      std::array<std::size_t, sizeof...(SubElements)> forwardEdgeCount{};
      std::array<std::array<std::size_t, sizeof...(SubElements)>,
                 sizeof...(SubElements)>
          forwardEdgeList{};
      std::array<std::size_t, sizeof...(SubElements)> incomingEdgeCount{};
      for (auto &&[index, iter] :
           llvm::enumerate(llvm::ArrayRef(dependencies).drop_back())) {
        if (iter.getInputDependencies().empty() ||
            iter.getInputDependencies() == llvm::ArrayRef{PARENT_DEPENDENCY}) {
          // No dependency (besides the parent context which is satisfied).
          worklist[workListSize++] = index;
        } else {
          // Build the outgoing edge list but do keep track of the number of
          // incoming edges.
          for (auto fromIndex : iter.getInputDependencies())
            if (fromIndex != PARENT_DEPENDENCY) {
              forwardEdgeList[fromIndex][forwardEdgeCount[fromIndex]++] = index;
              ++incomingEdgeCount[index];
            }
        }
      }

      std::size_t topoOrderSize = 0;
      std::array<std::size_t, sizeof...(SubElements)> topoOrder;
      while (workListSize > 0) {
        std::size_t index = worklist[--workListSize];
        topoOrder[topoOrderSize++] = index;
        // "Remove" all outgoing edges from 'index'.
        // If a node has no more incoming edges add it to the worklist.
        for (auto &&m : llvm::ArrayRef(forwardEdgeList[index])
                            .take_front(forwardEdgeCount[index]))
          if (--incomingEdgeCount[m] == 0)
            worklist[workListSize++] = m;
      }

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
                // First generate the context for the subelement.
                context = dependencies[iter](subElements, contexts);
                // Now generate the subelement.
                std::get<index>(subElements) =
                    std::get<index>(std::make_tuple(generators...))(*context);
                // Check whether we were successful.
                return std::get<index>(subElements).has_value();
              }(indices) &&
                      ...);
            },
            getIndicesTuple(std::index_sequence_for<SubElements...>{}));

        // Discard this AST node if we failed to generate a subelement.
        if (!success)
          return std::nullopt;
      }
      // Lastly, generate the output context.
      std::get<sizeof...(SubElements)>(contexts) =
          dependencies[sizeof...(SubElements)](subElements, contexts);

      // And call the constructor with all subelements.
      // It should be safe to dereference all optionals since they have been
      // guaranteed to have been generated.
      return std::apply(
          [&](auto &&...values) { return constructor(std::move(*values)...); },
          std::move(subElements));
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
