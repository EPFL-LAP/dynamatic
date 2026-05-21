#ifndef DYNAMATIC_HLS_FUZZER_TYPE_SYSTEM_GUIDED_GENERATOR
#define DYNAMATIC_HLS_FUZZER_TYPE_SYSTEM_GUIDED_GENERATOR

#include "AST.h"
#include "Randomly.h"
#include "Utils.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FunctionExtras.h"

#include <any>

namespace dynamatic::gen {

/// Opaque wrapper which type-erases a context used during type checking.
/// It allows users of 'AbstractTypeSystem' to pass contexts around between
/// methods without needing to know the real context type used by the underlying
/// type system.
///
/// We call the type opaque since it does not implement any behavior based
/// on the contained context beyond being able to pass it around.
/// For an explanation of contexts, see the doc string for 'TypeSystem'.
class OpaqueContext {
public:
  template <
      typename TypingContext,
      std::enable_if_t<
          !std::is_same_v<OpaqueContext, std::decay_t<TypingContext>> &&
          !std::is_same_v<std::any, std::decay_t<TypingContext>>> * = nullptr>
  explicit OpaqueContext(TypingContext &&context)
      : container(std::forward<TypingContext>(context)) {}

  template <typename TypingContext>
  const TypingContext &cast() const {
    return *std::any_cast<TypingContext>(&container);
  }

  // Enable noop casts to 'OpaqueContext'.
  template <>
  const OpaqueContext &cast<OpaqueContext>() const {
    return *this;
  }

private:
  std::any container;
};

/// Sentinel value representing a dependency on the input context.
constexpr std::size_t INPUT_DEPENDENCY = -1;

/// Class responsible for telling the generator how to calculate the input
/// 'TypingContext' for a given subelement of 'ASTNode'.
/// The subelement whose input-context we are calculating for is given by its
/// position within 'TransferFnArray'. See that type definition for more
/// information.
///
/// The class allows specifying dependencies on previously calculated contexts
/// + previously generated subelements using 'inputIndices'.
/// The indices in 'inputIndices' refer to the index of the given subelement
/// this instance depends on within 'ASTNode::SubElements'.
/// The special value 'INPUT_DEPENDENCY' represents depending on the
/// input-context of 'ASTNode'.
/// It is the user's responsibility to not create cyclic dependencies.
template <typename TypingContext, typename ASTNode, std::size_t... inputIndices>
class TransferFn {

  template <typename Tuple, std::size_t current, std::size_t... remaining>
  struct CalcCompFn {
    // Recursive case.
    using type = typename CalcCompFn<
        decltype(std::tuple_cat(
            std::declval<Tuple>(),
            std::declval<std::conditional_t<
                current == INPUT_DEPENDENCY,
                // Input case, only add the context.
                std::tuple<const TypingContext &>,
                // Add both the context and the ASTNode to the arguments.
                std::tuple<
                    const TypingContext &,
                    const std::tuple_element_t<
                        std::min(current, std::tuple_size_v<
                                              typename ASTNode::SubElements> -
                                              1),
                        typename ASTNode::SubElements> &>>>())),
        remaining...>::type;
  };

  // Special case required to still allow input dependencies when 'ASTNode'
  // does not have any subelements.
  template <typename Tuple>
  struct CalcCompFn<Tuple, INPUT_DEPENDENCY, 0> {
    // Recursive case.
    using type = typename CalcCompFn<
        decltype(std::tuple_cat(
            std::declval<Tuple>(),
            std::declval<std::tuple<const TypingContext &>>())),
        0>::type;
  };

  // Terminating end-case
  template <class... Args, std::size_t current>
  struct CalcCompFn<std::tuple<Args...>, current> {
    using type = TypingContext(Args...);
  };

  using ContextComputationFn =
      typename CalcCompFn<std::tuple<>, inputIndices..., 0>::type;

public:
  /// Constructs a 'TransferFn' from a function.
  /// The signature of the function is dependent on 'inputIndices'.
  /// Specifically, for every element of 'inputIndices' and in the order as
  /// given in 'inputIndices', the arguments are:
  /// * The input 'TypingContext' if the value is 'INPUT_DEPENDENCY'
  /// * The output 'TypingContext' of the 'i'th subelement of 'ASTNode' followed
  ///   by the subelement's AST node itself.
  ///
  /// Example:
  /// Dependency<Context, ast::BinaryExpression,
  ///   /*rhs=*/ast::BINARY_EXPRESSION::RHS, INPUT_DEPENDENCY>(
  ///   [](const Context& rhsContext, const ast::Expression& rhs,
  ///      const Context& inputContext) -> Context {
  ///     ...
  ///   }
  /// )
  ///
  /// The function should always return a 'TypingContext'. All parameters are
  /// passed as const-references.
  explicit TransferFn(std::function<ContextComputationFn> computationFn)
      : computationFn(std::move(computationFn)) {}

  /// Convenience constructor from a constant 'TypingContext' without any
  /// dependencies.
  explicit TransferFn(TypingContext context)
      : TransferFn(
            [context = std::move(context)](auto &&...) { return context; }) {}

  template <typename... Args>
  TypingContext operator()(Args &&...args) const {
    return computationFn(std::forward<Args>(args)...);
  }

private:
  static_assert(((inputIndices <
                      std::tuple_size_v<typename ASTNode::SubElements> ||
                  inputIndices == INPUT_DEPENDENCY) &&
                 ...),
                "input indices must refer to subelements or the input");

  std::function<ContextComputationFn> computationFn;
};

/// Opaque-wrapper over 'TransferFn' that can be constructed from any instance
/// of 'TransferFn' with the same 'ASTNode'.
/// Users should construct 'TransferFn' instances instead.
///
/// Mainly used as a return type in 'AbstractTypeSystem' where templates cannot
/// or shouldn't be used.
template <typename ASTNode>
class OpaqueTransferFn {
  template <typename Tuple>
  struct OpaqueContextTupleImpl;

  template <typename... NonTerminals>
  struct OpaqueContextTupleImpl<std::tuple<NonTerminals...>> {
    using type = std::tuple<
        std::optional<std::conditional_t<true, OpaqueContext, NonTerminals>>...,
        std::optional<OpaqueContext>>;
  };

  template <typename Tuple>
  struct NonTerminalsTupleImpl;

  template <typename... NonTerminals>
  struct NonTerminalsTupleImpl<std::tuple<NonTerminals...>> {
    using type = std::tuple<std::optional<NonTerminals>...>;
  };

public:
  /// Tuple of optionals of all subelements of this ASTNode.
  /// This is used to have one consistent API with which to call an
  /// 'OpaqueDependency' to calculate a context.
  /// Elements are optional, since they may not yet have been constructed.
  using SubElementsTuple =
      typename NonTerminalsTupleImpl<typename ASTNode::SubElements>::type;

  /// Tuple of optionals of all contexts of this ASTNode.
  /// This is used to have one consistent API with which to call an
  /// 'OpaqueDependency' to calculate a context.
  /// Elements are optional, since they may not yet have been calculated.
  using ContextTuple =
      typename OpaqueContextTupleImpl<typename ASTNode::SubElements>::type;

  /// Constructs an 'OpaqueDependency' from a 'Dependency'.
  template <typename TypingContext, std::size_t... inputIndices>
  /*implicit*/ OpaqueTransferFn(
      TransferFn<TypingContext, ASTNode, inputIndices...> &&dep)
      : dep(std::move(dep)),
        computationFn(+[](const std::any &dep,
                          const SubElementsTuple &subElements,
                          const ContextTuple &contexts) -> OpaqueContext {
          // Construct a tuple of all arguments that 'dep' should be called
          // with.
          // This mainly uses 'inputIndices' to index into 'subElements' and
          // 'contexts'.
          // The logic here simply unwraps the optionals: It assumes that the
          // required contexts and subelements have already been generated.
          auto argTuple = std::tuple_cat([&](auto &&integral) {
            constexpr std::size_t index = decltype(integral){};
            if constexpr (index == INPUT_DEPENDENCY) {
              // Input context.
              return std::forward_as_tuple(
                  std::get<std::tuple_size_v<ContextTuple> - 1>(contexts)
                      ->template cast<TypingContext>());
            } else {
              // Subelement context + ASTNode.
              return std::forward_as_tuple(
                  std::get<index>(contexts)->template cast<TypingContext>(),
                  *std::get<index>(subElements));
            }
          }(std::integral_constant<std::size_t, inputIndices>{})...);

          return OpaqueContext(std::apply(
              *std::any_cast<
                  TransferFn<TypingContext, ASTNode, inputIndices...>>(&dep),
              std::move(argTuple)));
        }) {

    static std::array<std::size_t, sizeof...(inputIndices)> storage{
        inputIndices...};
    this->inputIndices = storage;
  }

  /// Returns the indices of the subelements (or input) that this dependency
  /// depends on.
  llvm::ArrayRef<std::size_t> getInputDependencies() const {
    return inputIndices;
  }

  /// Calculates the context from the currently calculated subelements and
  /// contexts. Internal API that should only be used by the generator.
  OpaqueContext operator()(const SubElementsTuple &subElements,
                           const ContextTuple &contexts) const {
    return computationFn(dep, subElements, contexts);
  }

private:
  std::any dep;
  OpaqueContext (*computationFn)(const std::any &dep,
                                 const SubElementsTuple &nonTerminals,
                                 const ContextTuple &tuple);
  llvm::ArrayRef<std::size_t> inputIndices;
};

/// Class responsible for calculating the output context after generating an
/// 'ASTNode' instance.
/// It primarily differs from 'TransferFn' in that it receives a fully
/// constructed instance of 'ASTNode' rather than subelements and is always
/// executed last.
/// There is no way for a 'TransferFn' to receive the final fully constructed
/// 'ASTNode' necessitating this being a separate class.
///
/// For example, given a 'TransferFn<ArrayAssignmentStatement, ...>' for an
/// array assignment of the form:
///   ARRAY[INDEX] = VALUE
///
/// If it its input dependencies are 'ArrayAssignmentStatement::ARRAY',
/// 'ArrayAssignmentStatement::NAME' and 'INPUT_CONTEXT' (i.e.
/// TransferFn<ArrayAssignmentStatement, ARRAY, NAME, INPUT_CONTEXT> in C++)
/// then its function object has the signature:
///
/// (const TypingContext& arrayContext, const ArrayParameter& arrayParameter,
///  const TypingContext& indexContext, const Expression& index,
///  const TypingContext& inputContext) -> TypingContext
///
/// An 'OutputTransferFn' with the same input context, in contrast, no longer
/// receives the AST subelements but the final generated 'ASTNode' as first
/// parameter. The function object for the above must have the signature:
///
/// (const ArrayAssignmentStatement& node,
///  const TypingContext& arrayContext, const TypingContext& indexContext,
///  const TypingContext& inputContext) -> TypingContext
template <typename ASTNode>
class OutputTransferFn {
public:
  /// Constructs a 'OutputTransferFn' from a function object and
  /// 'inputDependencies'.
  /// Like in 'TransferFn', the 'inputDependencies' specify the output contexts
  /// of the corresponding subelements that should be passed into the function
  /// object.
  /// The function object is expected to have the signature:
  ///
  /// TypingContext(const ASTNode& node, const TypingContext&...)
  ///
  /// where the typing contexts after the ast node correspond to the output
  /// contexts of the subelements.
  /// Note that unlike 'TransferFn', no subelement AST nodes are passed.
  /// Instead the fully constructed 'ASTNode' is passed as the first parameter.
  template <std::size_t... inputDependencies, class F>
  explicit OutputTransferFn(std::index_sequence<inputDependencies...>, F &&f)
      : computationFn([f = std::forward<F>(f)](
                          const ASTNode &astNode,
                          const typename OpaqueTransferFn<ASTNode>::ContextTuple
                              &contexts) {
          // Using the values in 'inputDependencies', construct a tuple of all
          // the requested contexts and unbox them out of the 'OpaqueContext'.
          auto castedContexts = enumerateTuplesInto(
              [](auto &&...args) {
                return std::forward_as_tuple(
                    std::forward<decltype(args)>(args)...);
              },
              [&](auto indexT, auto inputDependencyT) -> decltype(auto) {
                constexpr std::size_t index = decltype(indexT){};
                constexpr std::size_t inputDependency =
                    decltype(inputDependencyT){};

                // Input dependencies map to the last element within
                // 'contexts'.
                constexpr std::size_t indexInContext =
                    inputDependency == INPUT_DEPENDENCY
                        ? std::tuple_size_v<std::decay_t<decltype(contexts)>> -
                              1
                        : inputDependency;

                // Cast the 'OpaqueContext' to whatever parameter type
                // the function object accepts.
                constexpr std::size_t astNodeParamOffset = 1;

                using FunctionTrait =
                    llvm::function_traits<std::decay_t<decltype(f)>>;
                using TypingContext =
                    std::decay_t<typename FunctionTrait::template arg_t<
                        astNodeParamOffset + index>>;

                return std::get<indexInContext>(contexts)
                    ->template cast<TypingContext>();
              },
              getTupleOfIndices(std::index_sequence<inputDependencies...>{}));

          // Now call the given function object using the 'ASTNode' and the
          // contexts.
          return std::apply(
              [&](auto &&...args) {
                return OpaqueContext(
                    f(astNode, std::forward<decltype(args)>(args)...));
              },
              std::move(castedContexts));
        }) {}

  /// Convenience overload for 'OutputTransferFn' that do not have any input
  /// dependencies.
  template <class F, std::enable_if_t<std::is_invocable_v<F, const ASTNode &>>
                         * = nullptr>
  explicit OutputTransferFn(F &&f)
      : OutputTransferFn(std::index_sequence<>{}, std::forward<F>(f)) {}

  /// Convenience method for 'OutputTransferFn' that always return the same
  /// value.
  template <class T>
  static OutputTransferFn outputConstant(T &&value) {
    return OutputTransferFn(
        [value = std::forward<T>(value)](const ASTNode &) { return value; });
  }

  /// Calculates the context from the new ASTNode and the contexts.
  /// Internal API that should only be used by the generator.
  OpaqueContext operator()(
      const ASTNode &astNode,
      const typename OpaqueTransferFn<ASTNode>::ContextTuple &contexts) const {
    return computationFn(astNode, contexts);
  }

private:
  std::function<OpaqueContext(
      const ASTNode &astNode,
      const typename OpaqueTransferFn<ASTNode>::ContextTuple &contexts)>
      computationFn;
};

namespace details {
template <typename ASTNode, typename Tuple = typename ASTNode::SubElements>
struct CalculateDependencyArray;

template <typename ASTNode, typename... SubElements>
struct CalculateDependencyArray<ASTNode, std::tuple<SubElements...>> {
  using type = std::tuple<
      std::conditional_t<true, OpaqueTransferFn<ASTNode>, SubElements>...,
      OutputTransferFn<ASTNode>>;
};
} // namespace details

/// Tuple of transfer functions returned by 'AbstractTypeSystem' for every
/// 'ASTNode'.
/// The tuple contains as many elements as there are subelements in 'ASTNode'
/// plus one.
/// The corresponding index in the tuple corresponds to the 'OpaqueTransferFn'
/// instance used to calculate the input context for that subelement.
/// The special last element in the tuple is a 'OutputTransferFn' that
/// calculates the output context for the 'ASTNode'.
template <typename ASTNode>
using TransferFnArray =
    typename details::CalculateDependencyArray<ASTNode>::type;

/// Abstract base class for all type systems. Users of a type system such as
/// the C generator use this interface in conjunction with 'OpaqueContext' to be
/// able to pass on contexts for generating AST elements without needing to know
/// about the concrete context type used by the type system.
///
/// Without this abstract interface, generators would need to be almost entirely
/// C++ templates instantiated with a type system instance.
///
/// While it is possible for a type system to directly inherit from
/// 'AbstractTypeSystem', implementing the various 'check*' methods would
/// require manual boxing and unboxing of 'OpaqueContext's to the
/// type system's 'TypingContext'.
///
/// The 'TypeSystem' base class below should be used instead to automate this by
/// overriding all the methods in  'AbstractTypeSystem' that box and unbox
/// 'OpaqueContext's and dispatch to corresponding (non-opaque) methods
/// in the derived class.
/// It also offers common and convenient default implementations of 'check*'
/// and 'discard*' methods.
class AbstractTypeSystem {
protected:
  /// Returns an instance of 'TransferFn' which simply forwards the context
  /// from the input to the subelement.
  template <typename ASTNode>
  static auto copyFromInput() {
    return copyFrom<ASTNode, INPUT_DEPENDENCY>();
  }

  /// Returns an instance of 'TransferFn' which forwards the context
  /// from the given index to the subelement.
  template <typename ASTNode, std::size_t index>
  static auto copyFrom() {
    return TransferFn<OpaqueContext, ASTNode, index>(
        [](const OpaqueContext &context, auto &&...) { return context; });
  }

  /// Returns a noop 'OutputTransferFn' that keeps the output context
  /// equal to the input context.
  template <typename ASTNode>
  static auto copyInputToOutput() {
    return copyToOutput<ASTNode, INPUT_DEPENDENCY>();
  }

  /// Returns a 'OutputTransferFn' whose output context will be equivalent to
  /// the output context of 'index' subelement.
  template <typename ASTNode, std::size_t index>
  static auto copyToOutput() {
    return OutputTransferFn<ASTNode>(
        std::index_sequence<index>{},
        [](const ASTNode &, const OpaqueContext &context) { return context; });
  }

public:
  virtual ~AbstractTypeSystem();

  virtual TransferFnArray<ast::Function> getFunctionTransferFns() {
    return {
        /*return type=*/copyFromInput<ast::Function>(),
        /*statement list=*/copyFromInput<ast::Function>(),
        /*return statement=*/copyFromInput<ast::Function>(),
        /*output=*/copyInputToOutput<ast::Function>(),
    };
  }

  virtual TransferFnArray<ast::ReturnStatement>
  getReturnStatementTransferFns() {
    return {
        /*return value=*/copyFromInput<ast::ReturnStatement>(),
        /*output=*/copyInputToOutput<ast::ReturnStatement>(),
    };
  }

  virtual bool discardScalarTypeOpaque(const ast::ScalarType &scalarType,
                                       const OpaqueContext &context) = 0;

  virtual TransferFnArray<ast::ScalarType> getScalarTypeTransferFns() {
    return /*output=*/copyInputToOutput<ast::ScalarType>();
  }

  virtual bool discardReturnTypeOpaque(const ast::ReturnType &,
                                       const OpaqueContext &context) = 0;

  virtual TransferFnArray<ast::ReturnType> getReturnTypeTransferFns() {
    return /*output=*/copyInputToOutput<ast::ReturnType>();
  }

  /// Returns true if the generator should discard this binary expression
  /// based on the given input context.
  virtual bool discardBinaryExpressionOpaque(ast::BinaryExpression::Op op,
                                             const OpaqueContext &context) = 0;

  virtual TransferFnArray<ast::BinaryExpression>
  getBinaryExpressionTransferFns(ast::BinaryExpression::Op op) {
    // Default implementation: Simply propagates the context to the subelements.
    return {/*lhs=*/copyFromInput<ast::BinaryExpression>(),
            /*rhs=*/copyFromInput<ast::BinaryExpression>(),
            /*output=*/copyInputToOutput<ast::BinaryExpression>()};
  }

  virtual bool discardUnaryExpressionOpaque(ast::UnaryExpression::Op op,
                                            const OpaqueContext &context) = 0;

  virtual TransferFnArray<ast::UnaryExpression>
  getUnaryExpressionTransferFns(ast::UnaryExpression::Op op) {
    return {
        /*operand=*/copyFromInput<ast::UnaryExpression>(),
        /*output=*/copyInputToOutput<ast::UnaryExpression>(),
    };
  }

  virtual bool discardVariableOpaque(const OpaqueContext &context) = 0;

  virtual TransferFnArray<ast::Variable> getVariableTransferFns() {
    return {
        /*parameter=*/copyFromInput<ast::Variable>(),
        /*output=*/copyInputToOutput<ast::Variable>(),
    };
  }

  virtual bool discardCastExpressionOpaque(const OpaqueContext &context) = 0;

  virtual TransferFnArray<ast::CastExpression> getCastExpressionTransferFns() {
    return {
        /*target type=*/copyFromInput<ast::CastExpression>(),
        /*operand=*/copyFromInput<ast::CastExpression>(),
        /*output=*/copyInputToOutput<ast::CastExpression>(),
    };
  }

  virtual bool
  discardConditionalExpressionOpaque(const OpaqueContext &context) = 0;

  virtual TransferFnArray<ast::ConditionalExpression>
  getConditionalExpressionTransferFns() {
    // Default implementation: Simply propagates the context to the
    // subelements.
    return {
        /*condition=*/copyFromInput<ast::ConditionalExpression>(),
        /*true value=*/copyFromInput<ast::ConditionalExpression>(),
        /*false value=*/copyFromInput<ast::ConditionalExpression>(),
        /*output=*/copyInputToOutput<ast::ConditionalExpression>(),
    };
  }

  virtual std::optional<ast::Constant>
  discardConstantOpaque(const ast::Constant &,
                        const OpaqueContext &context) = 0;

  virtual TransferFnArray<ast::Constant> getConstantTransferFns() {
    return /*output=*/copyInputToOutput<ast::Constant>();
  }

  virtual bool
  discardExistingScalarParameterOpaque(const ast::ScalarParameter &,
                                       const OpaqueContext &context) = 0;

  virtual TransferFnArray<ast::ExistingScalarParameter>
  getExistingScalarParameterTransferFns() {
    return {
        /*output=*/copyInputToOutput<ast::ExistingScalarParameter>(),
    };
  }

  virtual bool
  discardFreshScalarParameterOpaque(const OpaqueContext &context) = 0;

  virtual TransferFnArray<ast::ScalarParameter>
  getFreshScalarParameterTransferFns() {
    return {
        /*data type=*/copyFromInput<ast::ScalarParameter>(),
        /*output=*/copyInputToOutput<ast::ScalarParameter>(),
    };
  }

  virtual bool
  discardArrayReadExpressionOpaque(const OpaqueContext &context) = 0;

  virtual TransferFnArray<ast::ArrayReadExpression>
  getArrayReadExpressionTransferFns() {
    return {/*array parameter=*/copyFromInput<ast::ArrayReadExpression>(),
            /*index=*/copyFromInput<ast::ArrayReadExpression>(),
            /*output=*/copyInputToOutput<ast::ArrayReadExpression>()};
  }

  virtual bool
  discardExistingArrayParameterOpaque(const ast::ArrayParameter &,
                                      const OpaqueContext &context) = 0;

  virtual TransferFnArray<ast::ExistingArrayParameter>
  getExistingArrayParameterTransferFns() {
    return {
        /*output=*/copyInputToOutput<ast::ExistingArrayParameter>(),
    };
  }

  virtual bool
  discardFreshArrayParameterOpaque(const OpaqueContext &context) = 0;

  virtual TransferFnArray<ast::ArrayParameter>
  getFreshArrayParameterTransferFns() {
    return {
        /*element type=*/copyFromInput<ast::ArrayParameter>(),
        /*output=*/copyInputToOutput<ast::ArrayParameter>(),
    };
  }

  virtual bool
  discardArrayAssignmentStatementOpaque(const OpaqueContext &context) = 0;

  virtual TransferFnArray<ast::ArrayAssignmentStatement>
  getArrayAssignmentStatementTransferFns() {
    return TransferFnArray<ast::ArrayAssignmentStatement>{
        /*array parameter=*/copyFromInput<ast::ArrayAssignmentStatement>(),
        /*index=*/copyFromInput<ast::ArrayAssignmentStatement>(),
        /*value=*/copyFromInput<ast::ArrayAssignmentStatement>(),
        /*output=*/copyInputToOutput<ast::ArrayAssignmentStatement>(),
    };
  }

  virtual bool discardStatementListOpaque(const OpaqueContext &context) = 0;

  virtual TransferFnArray<ast::StatementList> getStatementListTransferFns() {
    return TransferFnArray<ast::StatementList>{
        /*statement list=*/copyFromInput<ast::StatementList>(),
        /*statement=*/copyFromInput<ast::StatementList>(),
        /*output=*/copyInputToOutput<ast::StatementList>(),
    };
  }
};

/// CRTP-Base class for all implementations of a type system.
/// See https://en.cppreference.com/w/cpp/language/crtp.html for an explanation
/// of CRTP.
/// The 'Self' template type parameter should be the class deriving from
/// 'TypeSystem'.
///
/// Type systems are used to "guide" the generator by 1) deriving new contexts
/// used when generating sub-elements of an AST-node or 2) rejecting AST-nodes
/// entirely based on the current type context.
///
/// All type checking is performed under a given context specified as the
/// 'TypingContext' template parameter. Every AST node is initially generated
/// using an input context passed into the 'discard*' method of the AST node
/// which may discard the AST node. Otherwise, new contexts for the subelements
/// of the AST node can be derived.
///
/// The transfer functions allow specifying how input contexts for AST
/// elements should be calculated.
/// Specifically, an instance of 'TransferFn' can specify that it depends on the
/// context and AST node of a sibling subelement in addition to, or instead of
/// the input context.
/// Example:
/// Given the C expression 'a[i]', an input context can be derived for
/// generating 'i' using knowledge gained from the output context and AST node
/// 'a'.
/// The generator uses this knowledge to generate the AST node of 'a' before
/// 'i'.
///
/// Note: We call it contexts rather than constraints to match literature, and
/// as it more generally informs an AST-node generation about the type-system
/// state rather than necessarily putting requirements on an AST-node
/// generation.
///
/// The logic that should be implemented can be thought of as inversions of the
/// usual type checking rules seen in literature.
/// E.g. assuming a type system where the context is a two-state variable that
/// requires the expression to either be an integer type or a floating point
/// type, then a typing rule for conditional expressions might look as follows:
///
/// {integer} |- cond   {A} |- lhs   {A} |- rhs
/// -------------------------------------------
///        {A} |- cond ? lhs : rhs
///
/// which can also be written as:
/// ({integer} |- cond) -> ({A} |- lhs) -> ({A} |- rhs) -> ({A} |- cond ? lhs :
/// rhs)
///
/// The corresponding 'getConditionalExpressionTransferFns' method
/// instead implements:
/// ({A} |- cond ? lhs : rhs) -> ({integer} |- cond) -> ({A} |- lhs) -> ({A} |-
/// rhs) where 'A' is the input context and the three clauses correspond to the
/// input contexts of the sub elements.
///
/// The current implementation how a type system is used in the base generator
/// has a few constraints:
/// * For any given context, it must always be possible to generate some
///   expression, otherwise the generator loops forever.
/// * For any given context, it must always be possible to generate a function
///   return type.
template <typename TypingContext, typename Self>
class TypeSystem : public AbstractTypeSystem {
public:
  template <typename ASTNode, std::size_t... inputIndices>
  using TransferFn = TransferFn<TypingContext, ASTNode, inputIndices...>;

  /// Shorthand for derived classes to be able to call the default
  /// implementation of methods.
  using Super = TypeSystem;

  // Methods that can be overwritten in subclasses. Note these are not virtual
  // since we use CRTP-techniques to call these. They may be but are not
  // required to be static.

  static bool discardBinaryExpression(ast::BinaryExpression::Op,
                                      const TypingContext &) {
    return false;
  }

  static bool discardUnaryExpression(ast::UnaryExpression::Op,
                                     const TypingContext &) {
    return false;
  }

  static bool discardVariable(const TypingContext &) { return false; }

  static bool discardCastExpression(const TypingContext &) { return false; }

  static bool discardConditionalExpression(const TypingContext &) {
    return false;
  }

  static bool discardScalarType(const ast::ScalarType &,
                                const TypingContext &) {
    return false;
  }

  bool discardReturnType(const ast::ReturnType &returnType,
                         const TypingContext &context) {
    // Default implementation dispatches to 'checkScalarType'.
    return llvm::TypeSwitch<ast::ReturnType, bool>(returnType)
        .Case([](const ast::VoidType *) { return false; })
        .Case([&](const ast::ScalarType *scalar) {
          return self().discardScalarType(*scalar, context);
        });
  }

  std::optional<ast::Constant> discardConstant(const ast::Constant &constant,
                                               const TypingContext &context) {
    if (self().discardScalarType(constant.getType(), context))
      return std::nullopt;

    return constant;
  }

  bool discardExistingScalarParameter(const ast::ScalarParameter &parameter,
                                      const TypingContext &context) {
    return self().discardScalarType(parameter.getDataType(), context);
  }

  static bool discardFreshScalarParameter(const TypingContext &) {
    return false;
  }

  static bool discardArrayReadExpression(const TypingContext &) {
    return false;
  }

  bool discardExistingArrayParameter(const ast::ArrayParameter &parameter,
                                     const TypingContext &context) {
    return self().discardScalarType(parameter.getElementType(), context);
  }

  static bool discardFreshArrayParameter(const TypingContext &) {
    return false;
  }

  static bool discardArrayAssignmentStatement(const TypingContext &) {
    return false;
  }

  static bool discardStatementList(const TypingContext &) { return false; }

  // Implementations of the virtual methods in 'AbstractTypeSystem'.
  // These are automatically implemented to unbox the 'TypingContext's out of
  // the opaque contexts, calling the corresponding non-opaque 'check*' method
  // and boxing the result into an opaque context again.

  bool discardBinaryExpressionOpaque(ast::BinaryExpression::Op op,
                                     const OpaqueContext &context) final {
    return self().discardBinaryExpression(op, context.cast<TypingContext>());
  }

  bool discardUnaryExpressionOpaque(ast::UnaryExpression::Op op,
                                    const OpaqueContext &context) final {
    return self().discardUnaryExpression(op, context.cast<TypingContext>());
  }

  bool discardVariableOpaque(const OpaqueContext &context) final {
    return self().discardVariable(context.cast<TypingContext>());
  }

  bool discardCastExpressionOpaque(const OpaqueContext &context) final {
    return self().discardCastExpression(context.cast<TypingContext>());
  }

  bool discardConditionalExpressionOpaque(const OpaqueContext &context) final {
    return self().discardConditionalExpression(context.cast<TypingContext>());
  }

  bool discardScalarTypeOpaque(const ast::ScalarType &node,
                               const OpaqueContext &context) final {
    return self().discardScalarType(node, context.cast<TypingContext>());
  }

  bool discardReturnTypeOpaque(const ast::ReturnType &node,
                               const OpaqueContext &context) final {
    return self().discardReturnType(node, context.cast<TypingContext>());
  }

  std::optional<ast::Constant>
  discardConstantOpaque(const ast::Constant &node,
                        const OpaqueContext &context) final {
    return self().discardConstant(node, context.cast<TypingContext>());
  }

  bool
  discardExistingScalarParameterOpaque(const ast::ScalarParameter &node,
                                       const OpaqueContext &context) final {
    return self().discardExistingScalarParameter(node,
                                                 context.cast<TypingContext>());
  }

  bool discardFreshScalarParameterOpaque(const OpaqueContext &context) final {
    return self().discardFreshScalarParameter(context.cast<TypingContext>());
  }

  bool discardArrayReadExpressionOpaque(const OpaqueContext &context) final {
    return self().discardArrayReadExpression(context.cast<TypingContext>());
  }

  bool discardExistingArrayParameterOpaque(const ast::ArrayParameter &node,
                                           const OpaqueContext &context) final {
    return self().discardExistingArrayParameter(node,
                                                context.cast<TypingContext>());
  }

  bool discardFreshArrayParameterOpaque(const OpaqueContext &context) final {
    return self().discardFreshArrayParameter(context.cast<TypingContext>());
  }

  bool
  discardArrayAssignmentStatementOpaque(const OpaqueContext &context) final {
    return self().discardArrayAssignmentStatement(
        context.cast<TypingContext>());
  }

  bool discardStatementListOpaque(const OpaqueContext &context) final {
    return self().discardStatementList(context.cast<TypingContext>());
  }

private:
  Self &self() { return static_cast<Self &>(*this); }

  const Self &self() const { return static_cast<const Self &>(*this); }
};

/// A noop-system which uses all the default implementations in 'TypeSystem'.
/// Puts no constraints onto the base generator.
class NoopTypeSystem final : public TypeSystem<std::monostate, NoopTypeSystem> {
public:
  ~NoopTypeSystem() override;
};

/// Convenience type system that disallows every AST constructs (besides
/// functions) by default.
template <typename TypingContext, typename Self>
class DisallowByDefaultTypeSystem : public TypeSystem<TypingContext, Self> {

public:
  static bool discardBinaryExpression(ast::BinaryExpression::Op,
                                      const TypingContext &) {
    return true;
  }

  static bool discardUnaryExpression(ast::UnaryExpression::Op,
                                     const TypingContext &) {
    return true;
  }

  static bool discardVariable(const TypingContext &) { return true; }

  static bool discardCastExpression(const TypingContext &) { return true; }

  static bool discardConditionalExpression(const TypingContext &) {
    return true;
  }

  static bool discardScalarType(const ast::ScalarType &,
                                const TypingContext &) {
    return true;
  }

  static bool discardReturnType(const ast::ReturnType &,
                                const TypingContext &) {
    return true;
  }

  std::optional<ast::Constant> discardConstant(const ast::Constant &,
                                               const TypingContext &) {
    return std::nullopt;
  }

  static bool discardExistingScalarParameter(const ast::ScalarParameter &,
                                             const TypingContext &) {
    return true;
  }

  static bool discardFreshScalarParameter(const TypingContext &) {
    return true;
  }

  static bool discardArrayReadExpression(const TypingContext &) { return true; }

  static bool discardExistingArrayParameter(const ast::ArrayParameter &,
                                            const TypingContext &) {
    return true;
  }

  static bool discardFreshArrayParameter(const TypingContext &) { return true; }

  static bool discardArrayAssignmentStatement(const TypingContext &) {
    return true;
  }

  static bool discardStatementList(const TypingContext &) { return true; }
};

} // namespace dynamatic::gen

#endif
