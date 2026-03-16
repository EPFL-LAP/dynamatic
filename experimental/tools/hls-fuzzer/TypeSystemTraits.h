#ifndef DYNAMATIC_HLS_FUZZER_TYPE_SYSTEM_TRAITS
#define DYNAMATIC_HLS_FUZZER_TYPE_SYSTEM_TRAITS

#include "AST.h"

#include <tuple>

namespace dynamatic {

/// Class that provides meta-information about how 'ASTNode' is used in the type
/// system.
/// Specializations should provide two definitions:
/// * constexpr static bool CAN_DISCARD: to denote whether a node is
/// discardable.
/// * template <typename TypingContext> ... Conclusion: The conclusion type of
///   the 'ASTNode' that can be instantiated with any typing contexts.
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
template <typename ASTNode, typename TypingContext>
using ConclusionOf =
    typename TypeSystemTraits<ASTNode>::template Conclusions<TypingContext>;

/// Conclusion type of 'ASTNode', possible wrapped in a 'std::optional' if the
/// node is discardable.
template <typename ASTNode, typename TypingContext,
          bool discardable = TypeSystemTraits<ASTNode>::CAN_DISCARD>
using MaybeConclusionOf = std::conditional_t<
    discardable,
    std::optional<typename TypeSystemTraits<ASTNode>::template Conclusions<
        TypingContext>>,
    typename TypeSystemTraits<ASTNode>::template Conclusions<TypingContext>>;

/// Struct implementing common defaults.
struct TypeSystemTraitsDefaults {
  constexpr static bool CAN_DISCARD = true;

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

  constexpr static bool CAN_DISCARD = false;
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
struct TypeSystemTraits<ast::Constant> : TypeSystemTraitsDefaults {};

template <>
struct TypeSystemTraits<ast::Parameter> : TypeSystemTraitsDefaults {

  /// Type constraint for constants that this parameter is initialized to during
  /// test bench generation.
  template <typename TypingContext>
  using Conclusions = TypingContext;
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
