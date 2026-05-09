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
  static bool discardScalarType(const ast::ScalarType &scalarType,
                                DynamaticTypingContext context);

  /// Discard 'op' based on the mode in 'context' and forward constraint to
  /// the operands as required.
  static bool discardBinaryExpression(ast::BinaryExpression::Op op,
                                      DynamaticTypingContext context);

  static bool discardUnaryExpression(ast::UnaryExpression::Op op,
                                     DynamaticTypingContext context);

  TransferFnArray<ast::UnaryExpression>
  getUnaryExpressionTransferFns(ast::UnaryExpression::Op op) override;

  TransferFnArray<ast::BinaryExpression>
  getBinaryExpressionTransferFns(ast::BinaryExpression::Op op) final;

  TransferFnArray<ast::ConditionalExpression>
  getConditionalExpressionTransferFns() override {
    return {
        /*condition=*/TransferFn<ast::ConditionalExpression>(
            DynamaticTypingContext{
                random.fromEnum<DynamaticTypingContext::Constraint>()}),
        /*true value=*/copyFromParent<ast::ConditionalExpression>(),
        /*false value=*/copyFromParent<ast::ConditionalExpression>(),
        /*output=*/copyFromParent<ast::ConditionalExpression>(),
    };
  }

  TransferFnArray<ast::ArrayReadExpression>
  getArrayReadExpressionTransferFns() final {
    return {/*array parameter=*/copyFromParent<ast::ArrayReadExpression>(),
            /*index=*/
            TransferFn<ast::ArrayReadExpression>(DynamaticTypingContext{
                DynamaticTypingContext::IntegerRequired}),
            /*output=*/copyFromParent<ast::ArrayReadExpression>()};
  }

  TransferFnArray<ast::ArrayAssignmentStatement>
  getArrayAssignmentStatementTransferFns() override {
    return {
        /*array parameter=*/copyFromParent<ast::ArrayAssignmentStatement>(),
        /*index=*/
        TransferFn<ast::ArrayAssignmentStatement>(
            DynamaticTypingContext{DynamaticTypingContext::IntegerRequired}),
        /*value=*/copyFromParent<ast::ArrayAssignmentStatement>(),
        /*output=*/copyFromParent<ast::ArrayAssignmentStatement>(),
    };
  }

private:
  Randomly &random;
};

} // namespace dynamatic::gen

#endif
