#ifndef HLS_FUZZER_TARGETS_HLSTYPESYSTEM
#define HLS_FUZZER_TARGETS_HLSTYPESYSTEM

#include "../TypeSystem.h"

namespace dynamatic::gen {

/// Typing context that used to avoid casts between floats and integers.
struct DynamaticTypingContext {
  enum Constraint {
    /// Expression must be of a floating-point type.
    FloatRequired,
    /// Expression must be of an integer type.
    IntegerRequired,
    MAX_VALUE = IntegerRequired,
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
  static bool discardBinaryExpression(ast::BinaryExpression::Op op,
                                      DynamaticTypingContext context);

  std::optional<ConclusionOf<ast::UnaryExpression>>
  checkUnaryExpression(ast::UnaryExpression::Op op,
                       DynamaticTypingContext context) const;

  DependencyArray<ast::BinaryExpression>
  getBinaryExpressionContextDependencies(ast::BinaryExpression::Op op) final;

  ConclusionOf<ast::ConditionalExpression>
  checkConditionalExpression(DynamaticTypingContext context) const {
    // Condition can be either a floating point type or integer type.
    // Either converts to a bool type without issues.
    return ConclusionOf<ast::ConditionalExpression>{
        {random.fromEnum<DynamaticTypingContext::Constraint>()},
        context,
        context,
    };
  }

  DependencyArray<ast::ArrayReadExpression>
  getArrayReadExpressionContextDependencies() final {
    return DependencyArray<ast::ArrayReadExpression>{
        copyFromParent<ast::ArrayReadExpression>(),
        Dependency<ast::ArrayReadExpression>([] {
          return DynamaticTypingContext{
              DynamaticTypingContext::IntegerRequired};
        }),
        copyFromParent<ast::ArrayReadExpression>()};
  }

  static std::optional<ConclusionOf<ast::ArrayAssignmentStatement>>
  checkArrayAssignmentStatement(DynamaticTypingContext context) {
    return ConclusionOf<ast::ArrayAssignmentStatement>{
        // Forward the context to the array parameter as is.
        /*parameter=*/context,
        // Indexing expression must be an integer.
        /*index=*/
        DynamaticTypingContext{DynamaticTypingContext::IntegerRequired},
        /*value=*/context,
    };
  }

private:
  Randomly &random;
};

} // namespace dynamatic::gen

#endif
