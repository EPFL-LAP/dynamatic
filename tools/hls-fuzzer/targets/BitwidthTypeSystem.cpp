#include "BitwidthTypeSystem.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "bitwidth-type-system"

auto dynamatic::gen::BitwidthTypeSystem::checkScalarType(
    const ast::ScalarType &scalarType, const BitwidthTypingContext &context)
    -> std::optional<ConclusionOf<ast::ScalarType>> {
  if (scalarType == ast::PrimitiveType::Double ||
      scalarType == ast::PrimitiveType::Float)
    return std::nullopt;

  // Only allow a datatype if either: We have no bitwidth requirement OR
  // the type restricts it to fit in the given bitwidth.
  if (std::optional<std::uint8_t> req = context.bitwidthRequirementOrNone();
      !req || *req >= scalarType.getBitwidth())
    return ConclusionOf<ast::ScalarType>{};

  return std::nullopt;
}

auto dynamatic::gen::BitwidthTypeSystem::checkConstant(
    const ast::Constant &constant, const BitwidthTypingContext &context) const
    -> std::optional<ConclusionOf<ast::Constant>> {
  // Allow all integer constants as we manually truncate them
  // (regardless of their C++ type).
  if (!checkScalarType(constant.getType(), ResultIsTruncated{}))
    return std::nullopt;

  // Any integer constant is okay.
  std::optional<std::uint8_t> req = context.bitwidthRequirementOrNone();
  if (!req)
    return ConclusionOf<ast::Constant>{};

  // Otherwise restrain it to our bitwidth.
  return std::visit(
      [&](auto &&value) -> ast::Constant {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_integral_v<T>) {
          // TODO: This is basically always a non-negative number.
          //       Figure out when/if it is safe to produce a negative number
          //       here!
          return {static_cast<T>(value & (1LL << *req) - 1)};
        }
        llvm_unreachable("double and float handled above");
      },
      constant.value);
}

auto dynamatic::gen::BitwidthTypeSystem::checkBinaryExpression(
    ast::BinaryExpression::Op op, const BitwidthTypingContext &context) const
    -> std::optional<ConclusionOf<ast::BinaryExpression>> {
  switch (op) {
  case ast::BinaryExpression::BitAnd: {
    // Bitand is distributive: Sub-expressions can assume they are truncated
    // as well.
    std::optional<std::uint8_t> req = context.bitwidthRequirementOrNone();
    if (!req)
      return ConclusionOf<ast::BinaryExpression>{ResultIsTruncated{},
                                                 ResultIsTruncated{}};

    // Otherwise, one operand is constrained to of the given maximum bitwidth
    // while the other can assume it is being truncated.
    // The choice of whether the left or right-hand-side is constrained is
    // arbitrary.
    return ConclusionOf<ast::BinaryExpression>{
        ResultIsTruncated{}, getInterestingBitWidthInRange(*req)};
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
    if (context.resultIsTruncated())
      return ConclusionOf<ast::BinaryExpression>{ResultIsTruncated{},
                                                 ResultIsTruncated{}};

    // TODO: We can choose bitwidths for the left and right operands of these
    //       expressions here to fit a maximum bitwidth.
    return std::nullopt;

  case ast::BinaryExpression::ShiftRight:
    // TODO: Figure out constraints here.
    return std::nullopt;
  case ast::BinaryExpression::Greater:
  case ast::BinaryExpression::GreaterEqual:
  case ast::BinaryExpression::Less:
  case ast::BinaryExpression::LessEqual:
  case ast::BinaryExpression::Equal:
  case ast::BinaryExpression::NotEqual:
    // These operations consume all bits to produce its result, we cannot
    // leave it unconstrained, otherwise the input expressions must be done
    // with higher bitwidths.

    // C performs automatic promotion to 'int' for all data types that are
    // smaller than 'int'. This might cause sign-extension in which case the
    // semantics are not equal to just performing the comparison at a given
    // bitwidth. If the comparison must be done with 'n' bits, the operands
    // then have to be computable with 'n - 1' bits.
    // We account for that by requiring one less bit than the global maximum
    // for the operands.
    // TODO: The sign-extension of the inputs is dependent on whether the type
    //       of the operands are signed or not. We could track this
    //       theoretically.
    if (globalMaxBitwidth == 1) {
      LLVM_DEBUG({
        llvm::dbgs()
            << "Discarding NotEqualExpression as the maximum global "
               "bitwidth == 1, which requires the comparison to be done "
               "on 0-bit integers (which does not exist in C)";
      });
      return std::nullopt;
    }

    return ConclusionOf<ast::BinaryExpression>{
        {getInterestingBitWidthInRange(globalMaxBitwidth - 1)},
        {getInterestingBitWidthInRange(globalMaxBitwidth - 1)}};

  case ast::BinaryExpression::BitOr:
  case ast::BinaryExpression::BitXor:
    // Distribute regarding truncation.
    return ConclusionOf<ast::BinaryExpression>{context, context};
  }
  llvm_unreachable("all enum cases handled");
}

auto dynamatic::gen::BitwidthTypeSystem::checkConditionalExpression(
    const BitwidthTypingContext &context) const
    -> ConclusionOf<ast::ConditionalExpression> {
  // The condition must be constrained to fit within the global max bitwidth.
  return {{getInterestingBitWidthInRange(globalMaxBitwidth)}, context, context};
}

auto dynamatic::gen::BitwidthTypeSystem::checkFunction(
    const BitwidthTypingContext &context) -> ConclusionOf<ast::Function> {
  // Return types are exempt from the bitwidth rules as they're an interface
  // type.
  // Any integer type is allowed in that case.
  return ConclusionOf<ast::Function>{
      /*returnType=*/ResultIsTruncated{},
      /*returnStatement=*/context,
  };
}

dynamatic::gen::BitwidthTypingContext
dynamatic::gen::BitwidthTypeSystem::getInterestingBitWidthInRange(
    uint8_t bitWidth) const {
  if (random.getRatherLowProbabilityBool())
    return BitwidthTypingContext(random.getInteger<std::uint32_t>(1, bitWidth));

  return BitwidthTypingContext(bitWidth);
}
