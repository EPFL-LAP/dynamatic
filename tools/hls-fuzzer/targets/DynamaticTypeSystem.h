#ifndef HLS_FUZZER_TARGETS_HLSTYPESYSTEM
#define HLS_FUZZER_TARGETS_HLSTYPESYSTEM

#include "../TypeSystem.h"

namespace dynamatic::gen {

/// Typing context that used to avoid casts between floats and integers.
struct DynamaticTypingContext {
  enum Constraint {
    /// Expression mustn't be of a floating-point type.
    FloatRequired,
    /// Expression mustn't be of an integer type.
    IntegerRequired,
    /// Expression can be whatever.
    None,
    MAX_VALUE = None,
  } constraint;
};

/// Custom type system that avoids expressions that dynamatic is known not to
/// be able to compile consistently.
///
/// These are currently 'float-to-int' casts and 'int-to-float' casts.
/// We therefore use a type system context to either disallow floats, integers
/// or neither depending on the context.
class DynamaticTypeSystem
    : public TypeSystem<DynamaticTypingContext, DynamaticTypeSystem> {
public:
  explicit DynamaticTypeSystem(Randomly &random) : random(random) {}

  /// Discard 'scalarType' based on the mode in 'context'.
  static std::optional<ConclusionOf<ast::ScalarType>>
  checkScalarType(const ast::ScalarType &scalarType,
                  DynamaticTypingContext context);

  /// Discard 'op' based on the mode in 'context' and forward constraint to
  /// the operands as required.
  std::optional<ConclusionOf<ast::BinaryExpression>>
  checkBinaryExpression(ast::BinaryExpression::Op op,
                        DynamaticTypingContext context) const;

  std::optional<ConclusionOf<ast::CastExpression>>
  checkCastExpression(DynamaticTypingContext context) {
    // Pick a specific constraints such that the 'to' type and the expression
    // are both integers or both floating point types.
    context = eliminateNone(context);
    return Super::checkCastExpression(context);
  }

  ConclusionOf<ast::Function> checkFunction(DynamaticTypingContext context) {
    context = eliminateNone(context);
    return Super::checkFunction(context);
  }

  static ConclusionOf<ast::ConditionalExpression>
  checkConditionalExpression(DynamaticTypingContext context) {
    // Condition can be either a floating point type or integer type.
    // Either converts to a bool type without issues.
    return ConclusionOf<ast::ConditionalExpression>{
        {DynamaticTypingContext::None},
        context,
        context,
    };
  }

private:
  /// If 'context' is none, randomly picks one of 'integer' or 'float'.
  DynamaticTypingContext eliminateNone(DynamaticTypingContext context) const;

  Randomly &random;
};

} // namespace dynamatic::gen

#endif
