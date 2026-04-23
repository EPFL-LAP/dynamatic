#ifndef HLS_FUZZER_TARGETS_BITWIDTHTYPESYSTEM
#define HLS_FUZZER_TARGETS_BITWIDTHTYPESYSTEM

#include "hls-fuzzer/TypeSystem.h"

namespace dynamatic::gen {

struct ResultIsTruncated {};

/// Typing context used for the bitwidth type system.
/// The context may be in one of two distinct states:
/// * There is a bitwidth requirement that requires that an expression can be
///   done at a given bitwidth or is otherwise illegal.
///
/// * An expression is allowed to assume that its result is truncated to some
///   arbitrary bitwidth 'b'. The expression is legal iff this assumption is
///   sufficient to perform computations with bitwidth 'b'.
///   This is commonly the case for operations that are distributive regarding
///   the truncation operation.
///
/// The second state is enabled when a bitand-expression is generated and
/// allows using more expressions and parameters as sub-expressions.
/// The type system uses these two states to guarantee that ALL computations in
/// the generated program can be performed at a bitwidth less-or-equal to a
/// value chosen randomly in the entry context.
class BitwidthTypingContext {
public:
  /// Constructs a 'BitwidthTypingContext' enabling the assumption that the
  /// result of an expression is truncated.
  /*implicit*/ BitwidthTypingContext(ResultIsTruncated)
      : variant(ResultIsTruncated{}) {}

  /// Constructs a 'BitwidthTypingContext' that puts a strict bitwidth
  /// requirement on the computation performed in an expression.
  explicit BitwidthTypingContext(std::uint8_t bitwidth) : variant(bitwidth) {}

  /// Returns true if it can be assumed that the result of the expression is
  /// truncated.
  bool resultIsTruncated() const {
    return std::holds_alternative<ResultIsTruncated>(variant);
  }

  /// Returns the strictly required bitwidth or an empty optional, if there is
  /// no strict bitwidth requirement.
  std::optional<std::uint8_t> bitwidthRequirementOrNone() const {
    const std::uint8_t *req = std::get_if<std::uint8_t>(&variant);
    if (!req)
      return std::nullopt;
    return *req;
  }

private:
  std::variant<ResultIsTruncated, std::uint8_t> variant;
};

class BitwidthTypeSystem
    : public TypeSystem<BitwidthTypingContext, BitwidthTypeSystem> {
public:
  explicit BitwidthTypeSystem(uint8_t globalMaxBitwidth, Randomly &random)
      : globalMaxBitwidth(globalMaxBitwidth), random(random) {}

  /// Disallows floats and doubles.
  static std::optional<ConclusionOf<ast::ScalarType>>
  checkScalarType(const ast::ScalarType &scalarType,
                  const BitwidthTypingContext &);

  std::optional<ConclusionOf<ast::ReturnType>>
  checkReturnType(const ast::ReturnType &returnType,
                  const BitwidthTypingContext &context) {
    // Disallow void.
    if (llvm::isa<ast::VoidType>(returnType))
      return std::nullopt;

    return Super::checkReturnType(returnType, context);
  }

  /// Disallow array-assignment statements.
  /// There is no rationale for doing so beyond the fact that we don't need
  /// them, since we can just generate expression trees, and that it makes
  /// synthesis faster.
  static std::optional<ConclusionOf<ast::ArrayAssignmentStatement>>
  checkArrayAssignmentStatement(const BitwidthTypingContext &) {
    return std::nullopt;
  }

  /// Forces constants to fit in the given bitwidth requirement.
  std::optional<ConclusionOf<ast::Constant>>
  checkConstant(const ast::Constant &constant,
                const BitwidthTypingContext &context) const;

  std::optional<ConclusionOf<ast::BinaryExpression>>
  checkBinaryExpression(ast::BinaryExpression::Op op,
                        const BitwidthTypingContext &context) const;

  ConclusionOf<ast::ConditionalExpression>
  checkConditionalExpression(const BitwidthTypingContext &context) const;

  static ConclusionOf<ast::Function>
  checkFunction(const BitwidthTypingContext &context);

private:
  /// Returns either 'bitWidth' or with a low probability, a value in the range
  /// [1, bitWidth].
  BitwidthTypingContext getInterestingBitWidthInRange(uint8_t bitWidth) const;

  uint8_t globalMaxBitwidth;
  Randomly &random;
};

} // namespace dynamatic::gen

#endif
