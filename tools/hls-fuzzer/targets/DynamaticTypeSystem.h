#ifndef HLS_FUZZER_TARGETS_HLSTYPESYSTEM
#define HLS_FUZZER_TARGETS_HLSTYPESYSTEM

#include "../TypeSystem.h"

namespace dynamatic::gen {

/// Typing context that used to avoid casts between floats and integers.
struct DynamaticTypingContext {
  enum Constraint {
    /// Expression may be either integer of float.
    Unconstrained,
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
class DynamaticTypeSystem final
    : public TypeSystem<DynamaticTypingContext, DynamaticTypeSystem> {
public:
  explicit DynamaticTypeSystem(Randomly &) {}

  TransferFnArray<ast::Function> getFunctionTransferFns() override {
    return {
        /*return type=*/copyFromInput<ast::Function>(),
        /*statement list=*/copyFromInput<ast::Function>(),
        // Return statement must match whatever type was calculated for the
        // return type.
        /*return statement=*/
        copyFrom<ast::Function, ast::Function::RETURN_TYPE>(),
        copyToOutput<ast::Function, ast::Function::RETURN_STATEMENT>(),
    };
  }

  TransferFnArray<ast::ReturnStatement>
  getReturnStatementTransferFns() override {
    return {
        /*return value=*/copyFromInput<ast::ReturnStatement>(),
        /*output=*/
        copyToOutput<ast::ReturnStatement,
                     ast::ReturnStatement::RETURN_VALUE>(),
    };
  }

  TransferFnArray<ast::ReturnType> getReturnTypeTransferFns() override {
    return {
        OutputTransferFn<ast::ReturnType>(
            [](const ast::ReturnType &returnType) -> DynamaticTypingContext {
              if (llvm::isa<ast::VoidType>(returnType))
                return DynamaticTypingContext{
                    DynamaticTypingContext::Unconstrained};

              return typeToContext(llvm::cast<ast::ScalarType>(returnType));
            }),
    };
  }

  /// Discard 'scalarType' based on the mode in 'context'.
  static bool discardScalarType(const ast::ScalarType &scalarType,
                                DynamaticTypingContext context);

  TransferFnArray<ast::ScalarType> getScalarTypeTransferFns() override {
    return {
        OutputTransferFn<ast::ScalarType>(
            [](const ast::ScalarType &scalarType) {
              return typeToContext(scalarType);
            }),
    };
  }

  /// Discard 'op' based on the mode in 'context' and forward constraint to
  /// the operands as required.
  static bool discardBinaryExpression(ast::BinaryExpression::Op op,
                                      DynamaticTypingContext context);

  TransferFnArray<ast::BinaryExpression>
  getBinaryExpressionTransferFns(ast::BinaryExpression::Op op) override;

  static bool discardUnaryExpression(ast::UnaryExpression::Op op,
                                     DynamaticTypingContext context);

  TransferFnArray<ast::UnaryExpression>
  getUnaryExpressionTransferFns(ast::UnaryExpression::Op op) override;

  TransferFnArray<ast::Variable> getVariableTransferFns() override {
    return {copyFromInput<ast::Variable>(),
            copyToOutput<ast::Variable, ast::Variable::PARAMETER>()};
  }

  TransferFnArray<ast::ScalarParameter>
  getFreshScalarParameterTransferFns() override {
    return {
        copyFromInput<ast::ScalarParameter>(),
        copyToOutput<ast::ScalarParameter, ast::ScalarParameter::DATATYPE>()};
  }

  TransferFnArray<ast::ExistingScalarParameter>
  getExistingScalarParameterTransferFns() override {
    return {OutputTransferFn<ast::ExistingScalarParameter>(
        [](const ast::ExistingScalarParameter &scalarParameter) {
          return typeToContext(scalarParameter.getDataType());
        })};
  }

  TransferFnArray<ast::ArrayParameter>
  getFreshArrayParameterTransferFns() override {
    return {
        copyFromInput<ast::ArrayParameter>(),
        copyToOutput<ast::ArrayParameter, ast::ArrayParameter::ELEMENT_TYPE>()};
  }

  TransferFnArray<ast::ExistingArrayParameter>
  getExistingArrayParameterTransferFns() override {
    return {OutputTransferFn<ast::ExistingArrayParameter>(
        [](const ast::ExistingArrayParameter &arrayParameter) {
          return typeToContext(arrayParameter.getElementType());
        })};
  }

  TransferFnArray<ast::ConditionalExpression>
  getConditionalExpressionTransferFns() override {
    return {
        TransferFn<ast::ConditionalExpression>(
            DynamaticTypingContext{DynamaticTypingContext::Unconstrained}),
        copyFromInput<ast::ConditionalExpression>(),
        // Forward choice of type made in the true expression into the false
        // expression.
        copyFrom<ast::ConditionalExpression,
                 ast::ConditionalExpression::TRUE_VAL>(),
        copyToOutput<ast::ConditionalExpression,
                     ast::ConditionalExpression::TRUE_VAL>(),
    };
  }

  TransferFnArray<ast::Constant> getConstantTransferFns() override {
    return {
        OutputTransferFn<ast::Constant>([](const ast::Constant &constant) {
          return typeToContext(constant.getType());
        }),
    };
  }

  TransferFnArray<ast::CastExpression> getCastExpressionTransferFns() override {
    return {
        copyFromInput<ast::CastExpression>(),
        // Whatever type was chosen for the target cast, its constraint must
        // be used for the expression.
        copyFrom<ast::CastExpression, ast::CastExpression::TARGET_TYPE>(),
        copyToOutput<ast::CastExpression, ast::CastExpression::TARGET_TYPE>(),
    };
  }

  TransferFnArray<ast::ArrayReadExpression>
  getArrayReadExpressionTransferFns() override {
    return {copyFromInput<ast::ArrayReadExpression>(),
            TransferFn<ast::ArrayReadExpression>(DynamaticTypingContext{
                DynamaticTypingContext::IntegerRequired}),
            copyToOutput<ast::ArrayReadExpression,
                         ast::ArrayReadExpression::ARRAY_PARAMETER>()};
  }

  TransferFnArray<ast::ArrayAssignmentStatement>
  getArrayAssignmentStatementTransferFns() override {
    return {
        TransferFn<ast::ArrayAssignmentStatement>(
            DynamaticTypingContext{DynamaticTypingContext::Unconstrained}),
        TransferFn<ast::ArrayAssignmentStatement>(
            DynamaticTypingContext{DynamaticTypingContext::IntegerRequired}),
        // Value constrained must match the array parameter!
        copyFrom<ast::ArrayAssignmentStatement,
                 ast::ArrayAssignmentStatement::ARRAY>(),
        copyInputToOutput<ast::ArrayAssignmentStatement>(),
    };
  }

  static bool discardStructuredForStatement(const DynamaticTypingContext &) {
    // TODO: Figure out how we want to handle non-termination.
    return true;
  }

private:
  /// Returns the given type constraint that matches the given 'scalarType'.
  static DynamaticTypingContext
  typeToContext(const ast::ScalarType &scalarType) {
    return llvm::TypeSwitch<ast::ScalarType, DynamaticTypingContext>(scalarType)
        .Case([](const ast::PrimitiveType *primitive) {
          if (primitive->isInteger())
            return DynamaticTypingContext{
                DynamaticTypingContext::IntegerRequired};

          return DynamaticTypingContext{DynamaticTypingContext::FloatRequired};
        });
  }
};

} // namespace dynamatic::gen

#endif
