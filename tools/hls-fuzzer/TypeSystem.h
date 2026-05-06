#ifndef DYNAMATIC_HLS_FUZZER_TYPE_SYSTEM_GUIDED_GENERATOR
#define DYNAMATIC_HLS_FUZZER_TYPE_SYSTEM_GUIDED_GENERATOR

#include "Randomly.h"
#include "TypeSystemTraits.h"

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

/// Sentinel value representing a dependency on the parent context.
constexpr std::size_t PARENT_DEPENDENCY = -1;

/// Class responsible for telling the generator how to calculate the input
/// 'TypingContext' for a given subelement of 'ASTNode'.
/// The subelement whose input-context we are calculating for is given by its
/// position within 'DependencyTuple'. See that type definition for more
/// information.
///
/// The class is called 'Dependency' as it allows specifying a dependency on
/// previously calculated contexts + previously generated subelements using
/// 'inputIndices'.
/// The indices in 'inputIndices' refer to the index of the given subelement
/// this instance depends on within 'TypeSystemTraits<ASTNode>::SubElements'.
/// The special value 'PARENT_DEPENDENCY' represents depending on the
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
                current == PARENT_DEPENDENCY,
                // Parent case, only add the context.
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

  // Terminating end-case
  template <class... Args, std::size_t current>
  struct CalcCompFn<std::tuple<Args...>, current> {
    using type = TypingContext(Args...);
  };

  using ContextComputationFn =
      typename CalcCompFn<std::tuple<>, inputIndices..., 0>::type;

public:
  /// Constructs a 'Dependency' from a function.
  /// The signature of the function is dependent on 'inputIndices'.
  /// Specifically, for every element of 'inputIndices' and in the order as
  /// given in 'inputIndices', the arguments are:
  /// * The parent 'TypingContext' if the value is 'PARENT_DEPENDENCY'
  /// * The output 'TypingContext' of the 'i'th subelement of 'ASTNode' followed
  ///   by the subelement's AST node itself.
  ///
  /// Example:
  /// Dependency<Context, ast::BinaryExpression, /*rhs*/ 1, PARENT_DEPENDENCY>(
  ///   [](const Context& rhsContext, const ast::Expression& rhs,
  ///      const Context& parentContext) -> Context {
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
                  inputIndices == PARENT_DEPENDENCY) &&
                 ...),
                "input indices must refer to subelements or the parent");

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
            if constexpr (index == PARENT_DEPENDENCY) {
              // Parent context.
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

  /// Returns the indices of the subelements (or parent) that this dependency
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

/// Array of transfer functions returned by 'AbstractTypeSystem' for every
/// 'ASTNode'.
/// The array contains as many elements as there are subelements in 'ASTNode'
/// plus one.
/// The corresponding index in the array corresponds to the 'OpaqueTransferFn'
/// instance used to calculate the input context for that subelement.
/// The special last element in the array corresponds to calculating the output
/// 'context' for the 'ASTNode'.
template <typename ASTNode>
using TransferFnArray =
    std::array<OpaqueTransferFn<ASTNode>,
               std::tuple_size_v<typename ASTNode::SubElements> + 1>;

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
  /// Returns an instance of 'TransferFn' which simply forwards the context from
  /// the parent to the subelement.
  template <typename ASTNode>
  static auto copyFromParent() {
    return TransferFn<OpaqueContext, ASTNode, PARENT_DEPENDENCY>(
        [](const OpaqueContext &context) { return context; });
  }

public:
  virtual ~AbstractTypeSystem();

  virtual ConclusionOf<ast::Function, OpaqueContext>
  checkFunctionOpaque(const OpaqueContext &context) = 0;

  /// Returns true if the generator should discard this binary expression based
  /// on the given input context.
  virtual bool discardBinaryExpressionOpaque(ast::BinaryExpression::Op op,
                                             const OpaqueContext &context) = 0;

  virtual std::optional<ConclusionOf<ast::UnaryExpression, OpaqueContext>>
  checkUnaryExpressionOpaque(ast::UnaryExpression::Op op,
                             const OpaqueContext &context) = 0;

  virtual TransferFnArray<ast::BinaryExpression>
  getBinaryExpressionContextDependencies(ast::BinaryExpression::Op op) {
    // Default implementation: Simply propagates the context to the subelements.
    return {/*lhs=*/copyFromParent<ast::BinaryExpression>(),
            /*rhs=*/copyFromParent<ast::BinaryExpression>(),
            /*output=*/copyFromParent<ast::BinaryExpression>()};
  }

  virtual std::optional<ConclusionOf<ast::Variable, OpaqueContext>>
  checkVariableOpaque(const OpaqueContext &context) = 0;

  virtual std::optional<ConclusionOf<ast::CastExpression, OpaqueContext>>
  checkCastExpressionOpaque(const OpaqueContext &context) = 0;

  virtual std::optional<ConclusionOf<ast::ConditionalExpression, OpaqueContext>>
  checkConditionalExpressionOpaque(const OpaqueContext &context) = 0;

  virtual std::optional<ConclusionOf<ast::ScalarType, OpaqueContext>>
  checkScalarTypeOpaque(const ast::ScalarType &,
                        const OpaqueContext &context) = 0;

  virtual std::optional<ConclusionOf<ast::ReturnType, OpaqueContext>>
  checkReturnTypeOpaque(const ast::ReturnType &,
                        const OpaqueContext &context) = 0;

  virtual std::optional<ConclusionOf<ast::Constant, OpaqueContext>>
  checkConstantOpaque(const ast::Constant &, const OpaqueContext &context) = 0;

  virtual std::optional<ConclusionOf<ast::ScalarParameter, OpaqueContext>>
  checkScalarParameterOpaque(const ast::ScalarParameter &,
                             const OpaqueContext &context) = 0;

  virtual bool
  discardArrayReadExpressionOpaque(const OpaqueContext &context) = 0;

  virtual TransferFnArray<ast::ArrayReadExpression>
  getArrayReadExpressionContextDependencies() {
    return {/*array parameter=*/copyFromParent<ast::ArrayReadExpression>(),
            /*index=*/copyFromParent<ast::ArrayReadExpression>(),
            /*output=*/copyFromParent<ast::ArrayReadExpression>()};
  }

  virtual std::optional<ConclusionOf<ast::ArrayParameter, OpaqueContext>>
  checkArrayParameterOpaque(const ast::ArrayParameter &,
                            const OpaqueContext &context) = 0;

  virtual std::optional<
      ConclusionOf<ast::ArrayAssignmentStatement, OpaqueContext>>
  checkArrayAssignmentStatementOpaque(const OpaqueContext &context) = 0;
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
/// There are currently two APIs to achieve this:
/// 1) The transfer functions API
/// 2) the 'check*' API.
/// The latter is considered deprecated and implements a subset of
/// functionality of the transfer functions API.
///
/// Regardless of API, all type checking is performed under a given context
/// specified as the 'TypingContext' template parameter.
/// Every AST node is initially generated using an input context
/// passed into the 'check*' method or 'discard*' method of the AST node which
/// may discard the AST node.
/// Otherwise, new contexts for the subelements of the AST node can be derived.
///
/// The transfer functions API allows specifying how input contexts for AST
/// elements should be calculated.
/// Specifically, an instance of 'TransferFn' can specify that it depends on the
/// context and AST node of a sibling subelement in addition to, or instead of
/// the parent input context.
/// Example:
/// Given the C expression 'a[i]', an input context can be derived for
/// generating 'i' using knowledge gained from the output context and AST node
/// 'a'.
/// The generator uses this knowledge to generate the AST node of 'a' before
/// 'i'.
/// 'check*' methods in contrast only implement deriving subelement contexts
/// from the parent input context.
/// They return a tuple of 'TypingContext's for every subelement of 'ASTNode',
/// the so-called conclusion type.
///
/// Note: We call it contexts rather than constraints to match literature, and
/// as it more generally informs an AST-node generation about the type-system
/// state rather than necessarily putting requirements on an AST-node
/// generation. In the future, it'll likely be possible to also output contexts
/// from sub-expressions to parent-expressions.
/// An example of such a context would e.g. be the set of all variables used.
///
/// The logic that should be implemented in the 'check*' methods can be thought
/// of as inversions of the usual type checking rules seen in literature.
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
/// The corresponding 'checkConditionalExpression' method instead implements:
/// ({A} |- cond ? lhs : rhs) -> ({integer} |- cond) -> ({A} |- lhs) -> ({A} |-
/// rhs) where 'A' is the context passed into the function and the three clauses
/// correspond to the conclusion type of conditional expressions.
///
/// Check methods for terminals are slightly special: They take as input an
/// already generated terminal node and are always discardable.
/// All check methods have a default implementation that forwards the current
/// constraint to all sub-elements.
/// See the 'TypeSystemTraits' specializations to find the documentation for
/// various AST-Node's conclusion types.
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
  /// The conclusion type of 'ASTNode' with the given context.
  template <typename ASTNode>
  using ConclusionOf = ConclusionOf<ASTNode, TypingContext>;

  template <typename ASTNode, std::size_t... inputIndices>
  using TransferFn = TransferFn<TypingContext, ASTNode, inputIndices...>;

  /// Shorthand for derived classes to be able to call the default
  /// implementation of methods.
  using Super = TypeSystem;

  // Methods that can be overwritten in subclasses. Note these are not virtual
  // since we use CRTP-techniques to call these. They may be but are not
  // required to be static.

  static ConclusionOf<ast::Function>
  checkFunction(const TypingContext &context) {
    return {context, context};
  }

  static bool discardBinaryExpression(ast::BinaryExpression::Op,
                                      const TypingContext &) {
    return false;
  }

  static ConclusionOf<ast::UnaryExpression>
  checkUnaryExpression(ast::UnaryExpression::Op, const TypingContext &context) {
    return {context};
  }

  static ConclusionOf<ast::Variable>
  checkVariable(const TypingContext &context) {
    return {context};
  }

  static ConclusionOf<ast::CastExpression>
  checkCastExpression(const TypingContext &context) {
    return {context, context};
  }

  static ConclusionOf<ast::ConditionalExpression>
  checkConditionalExpression(const TypingContext &context) {
    return {context, context, context};
  }

  static ConclusionOf<ast::ScalarType> checkScalarType(const ast::ScalarType &,
                                                       const TypingContext &) {
    return {};
  }

  std::optional<ConclusionOf<ast::ReturnType>>
  checkReturnType(const ast::ReturnType &returnType,
                  const TypingContext &context) {
    // Default implementation dispatches to 'checkScalarType'.
    return llvm::TypeSwitch<ast::ReturnType,
                            std::optional<ConclusionOf<ast::ReturnType>>>(
               returnType)
        .Case([](const ast::VoidType *) {
          return ConclusionOf<ast::ReturnType>{};
        })
        .Case([&](const ast::ScalarType *scalar)
                  -> std::optional<ConclusionOf<ast::ReturnType>> {
          if (std::optional optional = self().checkScalarType(*scalar, context);
              !optional)
            return std::nullopt;

          return ConclusionOf<ast::ReturnType>{};
        });
  }

  std::optional<ConclusionOf<ast::Constant>>
  checkConstant(const ast::Constant &constant, const TypingContext &context) {
    if (std::optional optional =
            self().checkScalarType(constant.getType(), context);
        !optional)
      return std::nullopt;

    return constant;
  }

  std::optional<ConclusionOf<ast::ScalarParameter>>
  checkScalarParameter(const ast::ScalarParameter &parameter,
                       const TypingContext &context) {
    if (std::optional optional =
            self().checkScalarType(parameter.getDataType(), context);
        !optional)
      return std::nullopt;

    return context;
  }

  static bool discardArrayReadExpression(const TypingContext &) {
    return false;
  }

  std::optional<ConclusionOf<ast::ArrayParameter>>
  checkArrayParameter(const ast::ArrayParameter &parameter,
                      const TypingContext &context) {
    if (std::optional optional =
            self().checkScalarType(parameter.getElementType(), context);
        !optional)
      return std::nullopt;

    return context;
  }

  static ConclusionOf<ast::ArrayAssignmentStatement>
  checkArrayAssignmentStatement(const TypingContext &context) {
    return {context, context, context};
  }

  // Implementations of the virtual methods in 'AbstractTypeSystem'.
  // These are automatically implemented to unbox the 'TypingContext's out of
  // the opaque contexts, calling the corresponding non-opaque 'check*' method
  // and boxing the result into an opaque context again.

  dynamatic::ConclusionOf<ast::Function, OpaqueContext>
  checkFunctionOpaque(const OpaqueContext &context) final {
    return convert(self().checkFunction(context.cast<TypingContext>()));
  }

  bool discardBinaryExpressionOpaque(ast::BinaryExpression::Op op,
                                     const OpaqueContext &context) final {
    return self().discardBinaryExpression(op, context.cast<TypingContext>());
  }

  std::optional<dynamatic::ConclusionOf<ast::UnaryExpression, OpaqueContext>>
  checkUnaryExpressionOpaque(ast::UnaryExpression::Op op,
                             const OpaqueContext &context) final {
    return convert(
        self().checkUnaryExpression(op, context.cast<TypingContext>()));
  }

  std::optional<dynamatic::ConclusionOf<ast::Variable, OpaqueContext>>
  checkVariableOpaque(const OpaqueContext &context) final {
    return convert(self().checkVariable(context.cast<TypingContext>()));
  }

  std::optional<dynamatic::ConclusionOf<ast::CastExpression, OpaqueContext>>
  checkCastExpressionOpaque(const OpaqueContext &context) final {
    return convert(self().checkCastExpression(context.cast<TypingContext>()));
  }

  std::optional<
      dynamatic::ConclusionOf<ast::ConditionalExpression, OpaqueContext>>
  checkConditionalExpressionOpaque(const OpaqueContext &context) final {
    return convert(
        self().checkConditionalExpression(context.cast<TypingContext>()));
  }

  std::optional<dynamatic::ConclusionOf<ast::ScalarType, OpaqueContext>>
  checkScalarTypeOpaque(const ast::ScalarType &node,
                        const OpaqueContext &context) final {
    return convert(self().checkScalarType(node, context.cast<TypingContext>()));
  }

  std::optional<dynamatic::ConclusionOf<ast::ReturnType, OpaqueContext>>
  checkReturnTypeOpaque(const ast::ReturnType &node,
                        const OpaqueContext &context) final {
    return convert(self().checkReturnType(node, context.cast<TypingContext>()));
  }

  std::optional<dynamatic::ConclusionOf<ast::Constant, OpaqueContext>>
  checkConstantOpaque(const ast::Constant &node,
                      const OpaqueContext &context) final {
    return convert(self().checkConstant(node, context.cast<TypingContext>()));
  }

  std::optional<dynamatic::ConclusionOf<ast::ScalarParameter, OpaqueContext>>
  checkScalarParameterOpaque(const ast::ScalarParameter &node,
                             const OpaqueContext &context) final {
    return convert(
        self().checkScalarParameter(node, context.cast<TypingContext>()));
  }

  bool discardArrayReadExpressionOpaque(const OpaqueContext &context) final {
    return self().discardArrayReadExpression(context.cast<TypingContext>());
  }

  std::optional<dynamatic::ConclusionOf<ast::ArrayParameter, OpaqueContext>>
  checkArrayParameterOpaque(const ast::ArrayParameter &node,
                            const OpaqueContext &context) final {
    return convert(
        self().checkArrayParameter(node, context.cast<TypingContext>()));
  }

  std::optional<
      dynamatic::ConclusionOf<ast::ArrayAssignmentStatement, OpaqueContext>>
  checkArrayAssignmentStatementOpaque(const OpaqueContext &context) final {
    return convert(
        self().checkArrayAssignmentStatement(context.cast<TypingContext>()));
  }

private:
  Self &self() { return static_cast<Self &>(*this); }

  const Self &self() const { return static_cast<const Self &>(*this); }

  static OpaqueContext convert(const TypingContext &context) {
    return OpaqueContext(context);
  }

  static OpaqueContext convert(TypingContext &&context) {
    return OpaqueContext(context);
  }

  template <class T>
  static auto convert(const T &value) {
    return value;
  }

  template <class T>
  static auto convert(std::optional<T> &&value) {
    using Ret = decltype(convert(std::move(*value)));
    if (!value)
      return std::optional<Ret>{};

    return std::optional<Ret>(convert(std::move(*value)));
  }

  /// Converts all instances of 'TypingContext' of a tuple-like struct into
  /// 'OpaqueContext'.
  /// Tuple-like structs are structs that specialize 'std::tuple_size' and
  /// implement 'get<size_t>' methods.
  template <template <typename> typename TupleLike,
            class Indices = std::make_index_sequence<
                std::tuple_size<std::decay_t<TupleLike<TypingContext>>>::value>>
  decltype(auto) convert(TupleLike<TypingContext> &&tuple) {
    return convertTupleLikeImpl(std::forward<TupleLike<TypingContext>>(tuple),
                                Indices{});
  }

  /// Converts all instances of 'TypingContext' in the tuple into
  /// 'OpaqueContext's.
  template <class... Args>
  static decltype(auto) convert(std::tuple<Args...> &&tuple) {
    return std::apply(
        [](auto &&...args) {
          return std::make_tuple(
              convert(std::forward<decltype(args)>(args))...);
        },
        std::move(tuple));
  }

  template <template <typename> typename TupleLike, std::size_t... indices>
  static decltype(auto) convertTupleLikeImpl(TupleLike<TypingContext> &&tuple,
                                             std::index_sequence<indices...>) {
    return TupleLike<OpaqueContext>{convert(get<indices>(tuple))...};
  }
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

  static std::optional<ConclusionOf<ast::UnaryExpression, TypingContext>>
  checkUnaryExpression(ast::UnaryExpression::Op, const TypingContext &) {
    return std::nullopt;
  }

  static std::optional<ConclusionOf<ast::Variable, TypingContext>>
  checkVariable(const TypingContext &) {
    return std::nullopt;
  }

  static std::optional<ConclusionOf<ast::CastExpression, TypingContext>>
  checkCastExpression(const TypingContext &) {
    return std::nullopt;
  }

  static std::optional<ConclusionOf<ast::ConditionalExpression, TypingContext>>
  checkConditionalExpression(const TypingContext &) {
    return std::nullopt;
  }

  static std::optional<ConclusionOf<ast::ScalarType, TypingContext>>
  checkScalarType(const ast::ScalarType &, const TypingContext &) {
    return std::nullopt;
  }

  static std::optional<ConclusionOf<ast::ReturnType, TypingContext>>
  checkScalarType(const ast::ReturnType &, const TypingContext &) {
    return std::nullopt;
  }

  std::optional<ConclusionOf<ast::Constant, TypingContext>>
  checkConstant(const ast::Constant &, const TypingContext &) {
    return std::nullopt;
  }

  std::optional<ConclusionOf<ast::ScalarParameter, TypingContext>>
  checkScalarParameter(const ast::ScalarParameter &, const TypingContext &) {
    return std::nullopt;
  }

  static bool discardArrayReadExpression(const TypingContext &) { return true; }

  std::optional<ConclusionOf<ast::ArrayParameter, TypingContext>>
  checkArrayParameter(const ast::ArrayParameter &, const TypingContext &) {
    return std::nullopt;
  }

  static std::optional<
      ConclusionOf<ast::ArrayAssignmentStatement, TypingContext>>
  checkArrayAssignmentStatement(const TypingContext &) {
    return std::nullopt;
  }
};

} // namespace dynamatic::gen

#endif
