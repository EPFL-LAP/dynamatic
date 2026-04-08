#ifndef DYNAMATIC_HLS_FUZZER_TYPE_SYSTEM_GUIDED_GENERATOR
#define DYNAMATIC_HLS_FUZZER_TYPE_SYSTEM_GUIDED_GENERATOR

#include "Randomly.h"
#include "TypeSystemTraits.h"

#include <any>

namespace dynamatic::gen {

/// Opaque wrapper which type-erases a context used during type checking.
/// It allows users of 'AbstractTypeSystem' to pass contexts returned by
/// 'check*' methods, to other 'check*' methods without needing to know the
/// real context type used by the underlying type system.
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

private:
  std::any container;
};

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
/// 'OpaqueContext's and dispatch to corresponding (non-opaque) 'check*' methods
/// in the derived class.
/// It also offers common and convenient default implementations of 'check*'
/// methods.
class AbstractTypeSystem {
public:
  virtual ~AbstractTypeSystem();

  /// Callback used by 'generate*' methods to generate subcomponents of an AST
  /// node.
  template <typename ASTNode, typename TypingContext = OpaqueContext>
  using GenerateCallback =
      llvm::function_ref<std::optional<ASTNode>(const TypingContext &context)>;

  virtual ConclusionOf<ast::Function, OpaqueContext>
  checkFunctionOpaque(const OpaqueContext &context) = 0;

  virtual std::optional<ConclusionOf<ast::BinaryExpression, OpaqueContext>>
  checkBinaryExpressionOpaque(ast::BinaryExpression::Op op,
                              const OpaqueContext &context) = 0;

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

  virtual std::optional<ConclusionOf<ast::ArrayReadExpression, OpaqueContext>>
  checkArrayReadExpressionOpaque(const OpaqueContext &context) = 0;

  virtual std::optional<ast::ArrayReadExpression>
  generateArrayReadExpressionOpaque(
      const OpaqueContext &context,
      GenerateCallback<ast::ArrayParameter> generateArrayParameter,
      GenerateCallback<ast::Expression> generateExpression) = 0;

  virtual std::optional<ConclusionOf<ast::ArrayParameter, OpaqueContext>>
  checkExistingArrayParameterOpaque(const ast::ArrayParameter &,
                                    const OpaqueContext &context) = 0;

  virtual std::optional<ast::ArrayParameter> generateFreshArrayParameterOpaque(
      const OpaqueContext &context,
      GenerateCallback<ast::ScalarType> generateScalarType,
      llvm::function_ref<std::string()> generateFreshVarName) = 0;

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
/// All type checking is performed under a given context specified as the
/// 'TypingContext' template parameter.
/// Two kinds of APIs exist to influence the generator:
/// * More high-level 'check*' methods for every kind of AST-node, which allow
///   deriving typing contexts for the sub-elements of the AST-node from the
///   input context, or rejecting the AST-node entirely.
///   The return type is the so-called conclusion and is different for every
///   AST construct. It is specified using the 'TypeSystemTraits'.
///
///   E.g. the conclusion of a binary expression are the contexts that should be
///   used to type check the left and right operands.
///   Most 'check*' methods support discarding the AST node entirely, in which
///   case the conclusion type is wrapped in an optional.
///
/// * Lower-level 'generate*' methods for AST-nodes, which allows customizing
///   the order in which sub-elements of AST-nodes are generated and deriving
///   typing contexts from properties of the generated AST-nodes.
///   The default implementations of 'generate*' methods perform left-to-right
///   generation (as it appears in C syntax) and derive typing contexts for
///   subelements using the corresponding 'check*' methods.
///
///   E.g. the generator function for an array-read expressions can be used to
///   first generate the array-parameter, and then use the dimension of the
///   array-parameter to derive a typing context that generates an in-bounds
///   expression for the indexing expression.
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
  explicit TypeSystem(Randomly &random) : random(random) {}

  /// The conclusion type of 'ASTNode' with the given context.
  template <typename ASTNode>
  using ConclusionOf = ConclusionOf<ASTNode, TypingContext>;

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

  static ConclusionOf<ast::BinaryExpression>
  checkBinaryExpression(ast::BinaryExpression::Op,
                        const TypingContext &context) {
    return {context, context};
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
          if (!self().checkScalarType(*scalar, context))
            return std::nullopt;

          return ConclusionOf<ast::ReturnType>{};
        });
  }

  std::optional<ConclusionOf<ast::Constant>>
  checkConstant(const ast::Constant &constant, const TypingContext &context) {
    if (!self().checkScalarType(constant.getType(), context))
      return std::nullopt;

    return constant;
  }

  std::optional<ConclusionOf<ast::ScalarParameter>>
  checkScalarParameter(const ast::ScalarParameter &parameter,
                       const TypingContext &context) {
    if (!self().checkScalarType(parameter.getDataType(), context))
      return std::nullopt;

    return context;
  }

  static ConclusionOf<ast::ArrayReadExpression>
  checkArrayReadExpression(const TypingContext &context) {
    return {context, context};
  }

  /// Default implementation for generating array-read expressions.
  /// Derives subelement typing contexts using 'checkArrayReadExpression'.
  /// The indexing expression is forced to be inbounds of the array parameter by
  /// using a bitmask. This requires array dimensions to be powers-of-2. This
  /// is guaranteed by the default implementation of
  /// 'generateFreshArrayParameter'.
  std::optional<ast::ArrayReadExpression> generateArrayReadExpression(
      const TypingContext &context,
      GenerateCallback<ast::ArrayParameter, TypingContext>
          generateArrayParameter,
      GenerateCallback<ast::Expression, TypingContext> generateExpression);

  std::optional<ConclusionOf<ast::ArrayParameter>>
  checkExistingArrayParameter(const ast::ArrayParameter &parameter,
                              const TypingContext &context) {
    if (!self().checkScalarType(parameter.getElementType(), context))
      return std::nullopt;

    return ConclusionOf<ast::ArrayParameter>{};
  }

  /// Default implementation for generating a new array parameter.
  /// Guarantees that a power-of-2 is used for the dimension of the array.
  std::optional<ast::ArrayParameter> generateFreshArrayParameter(
      const TypingContext &context,
      GenerateCallback<ast::ScalarType, TypingContext> generateScalarType,
      llvm::function_ref<std::string()> generateFreshVarName) {
    std::optional<ast::ScalarType> elementType = generateScalarType(context);
    if (!elementType)
      return std::nullopt;

    return ast::ArrayParameter{
        std::move(*elementType), generateFreshVarName(),
        // Generate a power-of-2 dimension to make the modulo operator fast and
        // easy to implement.
        // We choose an arbitrary upper-bound of 32 for the dimension for now.
        static_cast<std::size_t>(1 << random.getInteger(0, 5))};
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

  std::optional<dynamatic::ConclusionOf<ast::BinaryExpression, OpaqueContext>>
  checkBinaryExpressionOpaque(ast::BinaryExpression::Op op,
                              const OpaqueContext &context) final {
    return convert(
        self().checkBinaryExpression(op, context.cast<TypingContext>()));
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

  std::optional<
      dynamatic::ConclusionOf<ast::ArrayReadExpression, OpaqueContext>>
  checkArrayReadExpressionOpaque(const OpaqueContext &context) final {
    return convert(
        self().checkArrayReadExpression(context.cast<TypingContext>()));
  }

  std::optional<ast::ArrayReadExpression> generateArrayReadExpressionOpaque(
      const OpaqueContext &context,
      GenerateCallback<ast::ArrayParameter> generateArrayParameter,
      GenerateCallback<ast::Expression> generateExpression) final {
    return self().generateArrayReadExpression(convert(context),
                                              convert(generateArrayParameter),
                                              convert(generateExpression));
  }

  std::optional<dynamatic::ConclusionOf<ast::ArrayParameter, OpaqueContext>>
  checkExistingArrayParameterOpaque(const ast::ArrayParameter &node,
                                    const OpaqueContext &context) final {
    return convert(self().checkExistingArrayParameter(
        node, context.cast<TypingContext>()));
  }

  std::optional<ast::ArrayParameter> generateFreshArrayParameterOpaque(
      const OpaqueContext &context,
      GenerateCallback<ast::ScalarType> generateScalarType,
      llvm::function_ref<std::string()> generateFreshVarName) final {
    return convert(self().generateFreshArrayParameter(
        convert(context), convert(generateScalarType), generateFreshVarName));
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

  static const TypingContext &convert(const OpaqueContext &context) {
    return context.cast<TypingContext>();
  }

  template <class Ret, class... Args>
  static auto convert(llvm::function_ref<Ret(Args...)> function) {
    return [function](auto &&...args) {
      return convert(function(convert(std::forward<decltype(args)>(args)...)));
    };
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

protected:
  Randomly &random;
};

/// A noop-system which uses all the default implementations in 'TypeSystem'.
/// Puts no constraints onto the base generator.
class NoopTypeSystem : public TypeSystem<std::monostate, NoopTypeSystem> {
public:
  using TypeSystem::TypeSystem;
};

/// Convenience type system that disallows every AST constructs (besides
/// functions) by default.
template <typename TypingContext, typename Self>
class DisallowByDefaultTypeSystem : public TypeSystem<TypingContext, Self> {
public:
  using TypeSystem<TypingContext, Self>::TypeSystem;

  static std::optional<ConclusionOf<ast::BinaryExpression, TypingContext>>
  checkBinaryExpression(ast::BinaryExpression::Op, const TypingContext &) {
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

  static std::optional<ConclusionOf<ast::ArrayReadExpression, TypingContext>>
  checkArrayReadExpression(const TypingContext &) {
    return std::nullopt;
  }

  std::optional<ConclusionOf<ast::ArrayParameter, TypingContext>>
  checkExistingArrayParameter(const ast::ArrayParameter &,
                              const TypingContext &) {
    return std::nullopt;
  }

  static std::optional<
      ConclusionOf<ast::ArrayAssignmentStatement, TypingContext>>
  checkArrayAssignmentStatement(const TypingContext &) {
    return std::nullopt;
  }
};

template <typename TypingContext, typename Self>
std::optional<ast::ArrayReadExpression>
TypeSystem<TypingContext, Self>::generateArrayReadExpression(
    const TypingContext &context,
    GenerateCallback<ast::ArrayParameter, TypingContext> generateArrayParameter,
    GenerateCallback<ast::Expression, TypingContext> generateExpression) {
  std::optional conclusion = self().checkArrayReadExpression(context);
  if (!conclusion)
    return std::nullopt;
  auto [paramConc, indexConc] = *conclusion;

  std::optional<ast::ArrayParameter> arrayParameter =
      generateArrayParameter(paramConc);
  if (!arrayParameter)
    return std::nullopt;

  std::optional<ast::Expression> index = generateExpression(indexConc);
  if (!index)
    return std::nullopt;

  ast::ScalarType elementType = arrayParameter->getElementType();
  assert(llvm::isPowerOf2_64(arrayParameter->getDimension()) &&
         "default implementation depends on dimensions being powers of 2");
  std::size_t mask = arrayParameter->getDimension() - 1;
  std::string name = arrayParameter->getName().str();

  // Bitmask the index to be in range of the array! We use this to avoid
  // undefined behavior in our programs. In the future we could also add
  // mechanisms (type systems, or whatever), that restrict expressions to
  // safe in-range expressions.
  //
  // Note: We can use a bitmask here since array parameters that we generate
  // are all powers-of-2. We do so since the modulo operator is currently
  // unsupported in dynamatic.
  return ast::ArrayReadExpression{
      std::move(elementType), name,
      ast::BinaryExpression{std::move(*index), ast::BinaryExpression::BitAnd,
                            ast::Constant{static_cast<std::uint32_t>(mask)}}};
}

} // namespace dynamatic::gen

#endif
