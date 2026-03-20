#ifndef HLS_FUZZER_TARGETS_BITWIDTHTYPESYSTEM
#define HLS_FUZZER_TARGETS_BITWIDTHTYPESYSTEM

#include "hls-fuzzer/TypeSystem.h"

namespace dynamatic::gen {

/// Typing context used for the bitwidth type system.
struct BitwidthTypingContext {
  /// The upper-bound of required bitwidth for computations in a generated
  /// expression.
  /// More formally, "expr % (1 << maxBitwidth)" should be semantically
  /// identical to "expr" (local invariant).
  ///
  /// If this is an empty optional, the expression itself does not need to
  /// adhere to a specific bitwidth requirement but must either:
  /// * be distributive regarding the mod 2^n operator OR
  /// * satisfy the local invariant for some "maxBitwidth < globalMaxBitwidth".
  /// This property is applied recursively for any sub-expression in the first
  /// case.
  std::optional<std::uint8_t> maxBitwidth;
};

class BitwidthTypeSystem
    : public TypeSystem<BitwidthTypingContext, BitwidthTypeSystem> {
public:
  explicit BitwidthTypeSystem(uint8_t globalMaxBitwidth, Randomly &random)
      : globalMaxBitwidth(globalMaxBitwidth), random(random) {}

  /// Disallows floats and doubles.
  static std::optional<ConclusionOf<ast::ScalarType>>
  checkScalarType(const ast::ScalarType &scalarType,
                  const BitwidthTypingContext &) {
    if (scalarType == ast::PrimitiveType::Double ||
        scalarType == ast::PrimitiveType::Float)
      return std::nullopt;

    return ConclusionOf<ast::ScalarType>{};
  }

  std::optional<ConclusionOf<ast::Parameter>>
  checkParameter(const ast::Parameter &parameter,
                 const BitwidthTypingContext &context) {
    if (!Super::checkParameter(parameter, context))
      return std::nullopt;

    // Only allow a parameter if either: We have no bitwidth requirement OR
    // the parameter type restricts it to fit in the given bitwidth.
    if (!context.maxBitwidth ||
        *context.maxBitwidth >= parameter.getDataType().getBitwidth())
      return ConclusionOf<ast::Parameter>{};

    return std::nullopt;
  }

  // Forces constants to fit in the given bitwidth requirement.
  std::optional<ConclusionOf<ast::Constant>>
  checkConstant(const ast::Constant &constant,
                const BitwidthTypingContext &context);

  std::optional<ConclusionOf<ast::BinaryExpression>>
  checkBinaryExpression(ast::BinaryExpression::Op op,
                        const BitwidthTypingContext &context) const {
    switch (op) {
    case ast::BinaryExpression::BitAnd: {
      // Bitand is distributive: Sub-expressions can be unconstrained as well.
      if (!context.maxBitwidth)
        return ConclusionOf<ast::BinaryExpression>{context, context};

      // Otherwise, one operand is constrained to of the given maximum bitwidth
      // while the other can be unconstrained.
      // The choice of whether the left or right-hand-side is constrained is
      // arbitrary.
      return ConclusionOf<ast::BinaryExpression>{
          BitwidthTypingContext{std::nullopt},
          BitwidthTypingContext{getInterestingBitWidth(*context.maxBitwidth)}};
    }
    case ast::BinaryExpression::ShiftLeft:
      // TODO: Left shift is distributive for the shifted operand but not the
      //       shift-amount.
      //       Under a fixed bitwidth, we can also choose bitwidths for both
      //       operands such that it fits within a fixed bitwidth.
      return std::nullopt;

    case ast::BinaryExpression::Plus:
    case ast::BinaryExpression::Mul:
    case ast::BinaryExpression::Minus:
      if (!context.maxBitwidth)
        return ConclusionOf<ast::BinaryExpression>{context, context};

      // TODO: We can choose bitwidths for the left and right operands of these
      //       expressions here to fit a maximum bitwidth.
      return std::nullopt;

    case ast::BinaryExpression::ShiftRight:
    case ast::BinaryExpression::Greater:
    case ast::BinaryExpression::GreaterEqual:
    case ast::BinaryExpression::Less:
    case ast::BinaryExpression::LessEqual:
    case ast::BinaryExpression::Equal:
    case ast::BinaryExpression::NotEqual:
      // These operations consume all bits to produce its result, we cannot
      // leave it unconstrained, otherwise the input expressions must be done
      // with higher bitwidths.
      return ConclusionOf<ast::BinaryExpression>{
          {getInterestingBitWidth(globalMaxBitwidth)},
          {getInterestingBitWidth(globalMaxBitwidth)}};

    case ast::BinaryExpression::BitOr:
    case ast::BinaryExpression::BitXor:
      // Distribute regarding the mod operator.
      return ConclusionOf<ast::BinaryExpression>{context, context};
    }
    llvm_unreachable("all enum cases handled");
  }

  ConclusionOf<ast::ConditionalExpression>
  checkConditionalExpression(const BitwidthTypingContext &context) const {
    // The condition must be constrained to fit within the global max bitwidth.
    return {{getInterestingBitWidth(globalMaxBitwidth)}, context, context};
  }

private:
  /// Returns either 'bitWidth' or with a low probability, a value in the range
  /// [1, bitWidth].
  std::uint32_t getInterestingBitWidth(std::uint32_t bitWidth) const {
    if (random.getRatherLowProbabilityBool())
      return random.getInteger<std::uint32_t>(1, bitWidth);

    return bitWidth;
  }

  uint8_t globalMaxBitwidth;
  Randomly &random;
};

} // namespace dynamatic::gen

#endif
