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
  ///
  /// The empty optionals are introduced by bitand operations and mainly
  /// enables the use of parameters as well as computation done on higher
  /// bitwidths (such as addition or multiplication) to be capped.
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
                  const BitwidthTypingContext &);

  std::optional<ConclusionOf<ast::Parameter>>
  checkParameter(const ast::Parameter &parameter,
                 const BitwidthTypingContext &context);

  /// Forces constants to fit in the given bitwidth requirement.
  std::optional<ConclusionOf<ast::Constant>>
  checkConstant(const ast::Constant &constant,
                const BitwidthTypingContext &context);

  std::optional<ConclusionOf<ast::BinaryExpression>>
  checkBinaryExpression(ast::BinaryExpression::Op op,
                        const BitwidthTypingContext &context) const;

  ConclusionOf<ast::ConditionalExpression>
  checkConditionalExpression(const BitwidthTypingContext &context) const;

private:
  /// Returns either 'bitWidth' or with a low probability, a value in the range
  /// [1, bitWidth].
  std::uint32_t getInterestingBitWidthInRange(std::uint32_t bitWidth) const;

  uint8_t globalMaxBitwidth;
  Randomly &random;
};

} // namespace dynamatic::gen

#endif
