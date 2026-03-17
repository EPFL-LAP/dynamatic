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

/// Abstract base class for all type systems. Users of type systems should
/// use the methods here.
/// Implementations of type systems should derive from 'TypeSystem' which
/// implements all methods in this class.
class AbstractTypeSystem {
public:
  virtual ~AbstractTypeSystem();

#define DECLARE_NON_ATOMIC_OPAQUE_CHECK(ASTNode, methodName)                   \
  virtual MaybeConclusionOf<ASTNode, OpaqueContext> methodName##Opaque(        \
      const OpaqueContext &context) = 0

#define DECLARE_ATOMIC_OPAQUE_CHECK(ASTNode, methodName)                       \
  virtual MaybeConclusionOf<ASTNode, OpaqueContext> methodName##Opaque(        \
      const ASTNode &, const OpaqueContext &context) = 0

  DECLARE_NON_ATOMIC_OPAQUE_CHECK(ast::Function, checkFunction);

  virtual MaybeConclusionOf<ast::BinaryExpression, OpaqueContext>
  checkBinaryExpressionOpaque(ast::BinaryExpression::Op op,
                              const OpaqueContext &context) = 0;

  DECLARE_NON_ATOMIC_OPAQUE_CHECK(ast::Variable, checkVariable);
  DECLARE_NON_ATOMIC_OPAQUE_CHECK(ast::CastExpression, checkCastExpression);
  DECLARE_NON_ATOMIC_OPAQUE_CHECK(ast::ConditionalExpression,
                                  checkConditionalExpression);
  DECLARE_ATOMIC_OPAQUE_CHECK(ast::ScalarType, checkScalarType);
  DECLARE_ATOMIC_OPAQUE_CHECK(ast::Constant, checkConstant);
  DECLARE_ATOMIC_OPAQUE_CHECK(ast::Parameter, checkParameter);

#undef DECLARE_NON_ATOMIC_OPAQUE_CHECK
#undef DECLARE_ATOMIC_OPAQUE_CHECK
};

/// CRTP-Base class for all implementations of a type system.
/// See https://en.cppreference.com/w/cpp/language/crtp.html for an explanation
/// of CRTP.
/// The 'Self' template type parameter should be the class deriving from
/// 'TypeSystem'.
///
/// Type systems are used to "guide" the generator by either forwarding
/// constraints to sub-elements of an AST-node or rejecting AST-nodes entirely.
///
/// All type checking is performed under a given context specified as the
/// 'TypingContext' template parameter.
/// For every AST construct a corresponding 'check*' method exists.
/// The input to this method is always the context used to type check the given
/// AST construct.
/// The return type is the so-called conclusion and is different for every
/// AST construct. It is specified using the 'TypeSystemTraits'.
/// E.g. the conclusion of a binary expression are the contexts that should be
/// used to type check the left and right operands.
/// Most 'check*' methods support discarding the AST node entirely, in which
/// case the conclusion type is wrapped in an optional.
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
/// expression, otherwise the generator loops forever.
/// * For any given context, it must always be possible to generate some scalar
/// datatype, otherwise the generator loops forever.
template <typename TypingContext, class Self>
class TypeSystem : public AbstractTypeSystem {

public:
  // Implementation of methods in 'AbstractTypeSystem'.
#define IMPLEMENT_NON_TERMINAL_OPAQUE_CHECK(ASTNode, methodName)               \
  dynamatic::MaybeConclusionOf<ASTNode, OpaqueContext> methodName##Opaque(     \
      const OpaqueContext &context) final {                                    \
    return convert(self().methodName(context.cast<TypingContext>()));          \
  }                                                                            \
  static_assert(true, "forcing a semicolon")
#define IMPLEMENT_TERMINAL_OPAQUE_CHECK(ASTNode, methodName)                   \
  dynamatic::MaybeConclusionOf<ASTNode, OpaqueContext> methodName##Opaque(     \
      const ASTNode &node, const OpaqueContext &context) final {               \
    return convert(self().methodName(node, context.cast<TypingContext>()));    \
  }                                                                            \
  static_assert(true, "forcing a semicolon")

  IMPLEMENT_NON_TERMINAL_OPAQUE_CHECK(ast::Function, checkFunction);

  dynamatic::MaybeConclusionOf<ast::BinaryExpression, OpaqueContext>
  checkBinaryExpressionOpaque(ast::BinaryExpression::Op op,
                              const OpaqueContext &context) final {
    return convert(
        self().checkBinaryExpression(op, context.cast<TypingContext>()));
  }

  IMPLEMENT_NON_TERMINAL_OPAQUE_CHECK(ast::Variable, checkVariable);
  IMPLEMENT_NON_TERMINAL_OPAQUE_CHECK(ast::CastExpression, checkCastExpression);
  IMPLEMENT_NON_TERMINAL_OPAQUE_CHECK(ast::ConditionalExpression,
                                      checkConditionalExpression);
  IMPLEMENT_TERMINAL_OPAQUE_CHECK(ast::ScalarType, checkScalarType);
  IMPLEMENT_TERMINAL_OPAQUE_CHECK(ast::Constant, checkConstant);
  IMPLEMENT_TERMINAL_OPAQUE_CHECK(ast::Parameter, checkParameter);
#undef IMPLEMENT_NON_TERMINAL_OPAQUE_CHECK

  /// The conclusion type of 'ASTNode' with the given context.
  template <typename ASTNode>
  using ConclusionOf = ConclusionOf<ASTNode, TypingContext>;

  /// The conclusion type of 'ASTNode', possibly wrapped in a 'std::optional'
  /// if 'ASTNode' is discardable.
  template <typename ASTNode>
  using MaybeConclusionOf = MaybeConclusionOf<ASTNode, TypingContext>;

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

  static ConclusionOf<ast::Constant> checkConstant(const ast::Constant &,
                                                   const TypingContext &) {
    return {};
  }

  static ConclusionOf<ast::Parameter> checkParameter(const ast::Parameter &,
                                                     const TypingContext &) {
    return {};
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
class NoopTypeSystem : public TypeSystem<std::monostate, NoopTypeSystem> {};

} // namespace dynamatic::gen

#endif
