#ifndef DYNAMATIC_HLS_FUZZER_TARGETS_LSQNODEPTYPESYSTEM
#define DYNAMATIC_HLS_FUZZER_TARGETS_LSQNODEPTYPESYSTEM

#include "DynamaticTypeSystem.h"
#include "hls-fuzzer/ConjunctionTypeSystem.h"
#include "hls-fuzzer/LimitTypeSystem.h"
#include "hls-fuzzer/TypeSystem.h"

namespace dynamatic::gen {

struct LSQNoDepContext {
  /// The array parameter + index variable that is being written to in the
  /// current statement, or nullptr if not.
  const ast::ArrayParameter *arrayWritten = nullptr;
  const ast::Variable *indexVariable = nullptr;
  /// True iff the context is within the sub-tree of the indexing expression of
  /// an array read expression.
  bool inArrayReadIndexExpression = false;

  /// Returns true if the context indicates that we are currently generating an
  /// array index. This is either within an array-assignment statement or an
  /// array-read expression.
  ///
  /// Note that this is only accurate when generating an expression.
  bool generatingArrayIndex() const {
    // Note: !indexVariable is sufficient to tell we are in the index of an
    // array-assignment statement, merely because there is just one
    // array-assignment statement in the program at the moment,
    // the array parameter is not an expression and that the index variable
    // will be set when generating the value statement.
    return inArrayReadIndexExpression || !indexVariable;
  }
};

namespace detail {

/// Subtype system used for the LSQ related logic.
class LSQNoDepTypeSystemInner final
    : public TypeSystem<LSQNoDepContext, LSQNoDepTypeSystemInner> {
public:
  static bool discardReturnType(const ast::ReturnType &returnType,
                                const LSQNoDepContext &) {
    // Force a void return function.
    return returnType != ast::ReturnType{ast::VoidType{}};
  }

  static bool discardArrayReadExpression(const LSQNoDepContext &context) {
    // Anything but the index variable must be discarded when generating the
    // index.
    return context.generatingArrayIndex();
  }

  TransferFnArray<ast::ArrayReadExpression>
  getArrayReadExpressionTransferFns() override;

  TransferFnArray<ast::ArrayAssignmentStatement>
  getArrayAssignmentStatementTransferFns() override;

  static bool discardStructuredForStatement(const LSQNoDepContext &) {
    // Only array assignment statements should be allowed.
    return true;
  }

  static bool discardBinaryExpression(ast::BinaryExpression::Op,
                                      const LSQNoDepContext &context) {
    // Anything but the index variable must be discarded when generating the
    // index.
    return context.generatingArrayIndex();
  }

  static bool discardUnaryExpression(ast::UnaryExpression::Op,
                                     const LSQNoDepContext &context) {
    // Anything but the index variable must be discarded when generating the
    // index.
    return context.generatingArrayIndex();
  }

  static bool
  discardExistingScalarParameter(const ast::ScalarParameter &parameter,
                                 const LSQNoDepContext &context) {
    // In an indexing expression, the scalar parameter name must match the index
    // variable.
    if (!context.inArrayReadIndexExpression)
      return false;

    return parameter.getName() != context.indexVariable->name;
  }

  static bool discardFreshScalarParameter(const LSQNoDepContext &context) {
    return context.inArrayReadIndexExpression;
  }

  static bool
  discardExistingArrayParameter(const ast::ArrayParameter &parameter,
                                const LSQNoDepContext &context) {
    if (!context.arrayWritten)
      return false;

    return parameter.getName() != context.arrayWritten->getName();
  }

  static bool discardFreshArrayParameter(const LSQNoDepContext &context) {
    return context.arrayWritten;
  }

  static bool discardCastExpression(const LSQNoDepContext &context) {
    // Anything but the index variable must be discarded when generating the
    // index.
    return context.generatingArrayIndex();
  }

  static bool discardConditionalExpression(const LSQNoDepContext &context) {
    // Anything but the index variable must be discarded when generating the
    // index.
    return context.generatingArrayIndex();
  }

  std::optional<ast::Constant> discardConstant(const ast::Constant &constant,
                                               const LSQNoDepContext &context) {
    if (context.generatingArrayIndex())
      return std::nullopt;

    return TypeSystem::discardConstant(constant, context);
  }
};

} // namespace detail

/// Type system used to generate code that requires no LSQ to enforce ordering
/// constraints of memory.
/// Concretely, this means that:
/// * For any Write-after-Read (WAR), the write operation must be data dependent
///   on the read, guaranteed not to alias OR have a control dependency on the
///   read.
/// * For any Write-after-Write (WAW) or Read-after-Write (RAW) the operations
///   must not alias.
///
/// The current implementation only ever generates a single WAR construct where
/// the write is data dependent or control dependent on the read.
class LSQNoDepTypeSystem final
    : public ConjunctionTypeSystemBase<LSQNoDepTypeSystem,
                                       detail::LSQNoDepTypeSystemInner,
                                       DynamaticTypeSystem, LimitTypeSystem> {
public:
  explicit LSQNoDepTypeSystem()
      : ConjunctionTypeSystemBase(detail::LSQNoDepTypeSystemInner(),
                                  DynamaticTypeSystem(),
                                  LimitTypeSystem(/*maxExpressionDepth=*/8,
                                                  /*maxTotalStatements=*/1)) {}
};

} // namespace dynamatic::gen

#endif
