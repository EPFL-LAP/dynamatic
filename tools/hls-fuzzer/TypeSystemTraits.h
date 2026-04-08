#ifndef DYNAMATIC_HLS_FUZZER_TYPE_SYSTEM_TRAITS
#define DYNAMATIC_HLS_FUZZER_TYPE_SYSTEM_TRAITS

#include "AST.h"

#include <tuple>

namespace dynamatic {

/// Class that provides meta-information about how 'ASTNode' is used in the type
/// system.
/// Specializations should provide the following definitions:
///
/// * template <typename TypingContext> ... Conclusion:
///   The conclusion type of the 'ASTNode' that can be instantiated with any
///   typing contexts.
///
///   The conclusion type is the return type of the corresponding 'check*'
///   method of 'ASTNode' within 'TypeSystem' and usually contains contexts
///   that are used to generate the sub-elements of the 'ASTNode' and are
///   derived from the input context.
///
///   Templating the type allows it to be used for 'OpaqueContext's in the
///   'AbstractTypeSystem' as well as for any custom typing context type used
///   in a subclass of 'TypeSystem'.
///
///   The conclusion type of conditional expressions is e.g.
///   'std::tuple<TypingContext, TypingContext, TypingContext>' which are the
///   contexts used to type check the condition, true-expression and
///   false-expression respectively.
///
///   If this is a struct, it should be a tuple-like struct, i.e., specialize
///   'std::tuple_size' and implement a 'get<std::size_t>' method.
template <typename ASTNode>
struct TypeSystemTraits {
  // Always false.
  constexpr static bool NO_SPECIALIZATION = sizeof(ASTNode) == 0;

  static_assert(
      NO_SPECIALIZATION,
      "Missing specialization of 'TypeSystemTraits' for ASTNode"
      "Please add a specialization of 'TypeSystemTraits' for the given"
      " ASTNode");
};

/// Concrete conclusion type of 'ASTNode' with the given typing context.
/// Allows the C++ code to use a consistent typename to refer to conclusion
/// types, regardless of the 'ASTNode' used.
/// See the documentation of 'TypeSystemTraits::Conclusion'.
template <typename ASTNode, typename TypingContext>
using ConclusionOf =
    typename TypeSystemTraits<ASTNode>::template Conclusions<TypingContext>;

/// Struct implementing common defaults.
struct TypeSystemTraitsDefaults {
  /// No conclusion values forwarded. This is common for terminal nodes.
  template <typename>
  using Conclusions = std::tuple<>;
};

template <>
struct TypeSystemTraits<ast::Function> : TypeSystemTraitsDefaults {
  template <typename TypingContext>
  struct Conclusions {
    TypingContext returnType;
    TypingContext returnStatement;
  };
};

template <>
struct TypeSystemTraits<ast::Variable> : TypeSystemTraitsDefaults {

  /// Type constraint for the parameter this variable references or scalar
  /// types of fresh parameters created.
  template <typename TypingContext>
  using Conclusions = TypingContext;
};

template <>
struct TypeSystemTraits<ast::BinaryExpression> : TypeSystemTraitsDefaults {

  /// Type constraints for the left-hand operand followed by the right-hand
  /// operand.
  template <typename TypingContext>
  using Conclusions = std::tuple</*lhs=*/TypingContext, /*rhs=*/TypingContext>;
};

template <>
struct TypeSystemTraits<ast::CastExpression> : TypeSystemTraitsDefaults {

  /// Type constraints for the target type followed
  /// by the expression.
  template <typename TypingContext>
  using Conclusions =
      std::tuple</*type=*/TypingContext, /*expr=*/TypingContext>;
};

template <>
struct TypeSystemTraits<ast::ConditionalExpression> : TypeSystemTraitsDefaults {

  /// Type constraints for the condition followed by the true-expression
  /// followed by the false expression.
  template <typename TypingContext>
  using Conclusions =
      std::tuple</*condExpr=*/TypingContext, /*trueExpr=*/TypingContext,
                 /*falseExpr=*/TypingContext>;
};

template <>
struct TypeSystemTraits<ast::ScalarType> : TypeSystemTraitsDefaults {};

template <>
struct TypeSystemTraits<ast::ReturnType> : TypeSystemTraitsDefaults {};

template <>
struct TypeSystemTraits<ast::Constant> : TypeSystemTraitsDefaults {

  /// A possibly-modified constant that should be used instead by the generator.
  template <typename TypingContext>
  using Conclusions = ast::Constant;
};

template <>
struct TypeSystemTraits<ast::ScalarParameter> {

  /// Type constraint for constants that this parameter is initialized to during
  /// test bench generation.
  template <typename TypingContext>
  using Conclusions = TypingContext;
};

template <>
struct TypeSystemTraits<ast::ArrayReadExpression> {

  template <typename TypingContext>
  using Conclusions =
      std::tuple</*parameter=*/TypingContext, /*index=*/TypingContext>;
};

template <>
struct TypeSystemTraits<ast::ArrayParameter> : TypeSystemTraitsDefaults {};

template <>
struct TypeSystemTraits<ast::ArrayAssignmentStatement> {

  template <typename TypingContext>
  using Conclusions =
      std::tuple</*parameter=*/TypingContext, /*index=*/TypingContext,
                 /*value=*/TypingContext>;
};

} // namespace dynamatic

template <typename TypingContext>
struct std::tuple_size<dynamatic::TypeSystemTraits<
    dynamatic::ast::Function>::Conclusions<TypingContext>>
    : std::integral_constant<std::size_t, 2> {};

namespace dynamatic {
template <std::size_t i, typename TypingContext>
const TypingContext &
get(const ConclusionOf<ast::Function, TypingContext> &con) {
  static_assert(i < 2);
  if constexpr (i == 0) {
    return con.returnType;
  } else {
    return con.returnStatement;
  }
}
} // namespace dynamatic

#endif
