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

  virtual ConclusionOf<ast::Function, OpaqueContext>
  checkFunctionOpaque(const OpaqueContext &context) = 0;

  virtual std::optional<ConclusionOf<ast::BinaryExpression, OpaqueContext>>
  checkBinaryExpressionOpaque(ast::BinaryExpression::Op op,
                              const OpaqueContext &context) = 0;

  virtual std::optional<ConclusionOf<ast::UnaryExpression, OpaqueContext>>
  checkUnaryExpressionOpaque(ast::UnaryExpression::Op op,
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
/// All type checking is performed under a given context specified as the
/// 'TypingContext' template parameter.
/// For every AST construct a corresponding 'check*' method exists.
/// The input to this method is always the context used to type check the given
/// AST construct.
/// Based on the input context the 'check*' method can then derive new contexts
/// for its subelements or discard the AST-node entirely.
/// The return type is the so-called conclusion and is different for every
/// AST construct. It is specified using the 'TypeSystemTraits'.
///
/// E.g. the conclusion of a binary expression are the contexts that should be
/// used to type check the left and right operands.
/// Most 'check*' methods support discarding the AST node entirely, in which
/// case the conclusion type is wrapped in an optional.
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

  static ConclusionOf<ast::ArrayReadExpression>
  checkArrayReadExpression(const TypingContext &context) {
    return {context, context};
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

  std::optional<dynamatic::ConclusionOf<ast::BinaryExpression, OpaqueContext>>
  checkBinaryExpressionOpaque(ast::BinaryExpression::Op op,
                              const OpaqueContext &context) final {
    return convert(
        self().checkBinaryExpression(op, context.cast<TypingContext>()));
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

  std::optional<
      dynamatic::ConclusionOf<ast::ArrayReadExpression, OpaqueContext>>
  checkArrayReadExpressionOpaque(const OpaqueContext &context) final {
    return convert(
        self().checkArrayReadExpression(context.cast<TypingContext>()));
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
  static std::optional<ConclusionOf<ast::BinaryExpression, TypingContext>>
  checkBinaryExpression(ast::BinaryExpression::Op, const TypingContext &) {
    return std::nullopt;
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

  static std::optional<ConclusionOf<ast::ArrayReadExpression, TypingContext>>
  checkArrayReadExpression(const TypingContext &) {
    return std::nullopt;
  }

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
