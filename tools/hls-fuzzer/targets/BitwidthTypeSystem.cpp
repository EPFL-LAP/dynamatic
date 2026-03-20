#include "BitwidthTypeSystem.h"

auto dynamatic::gen::BitwidthTypeSystem::checkConstant(
    const ast::Constant &constant, const BitwidthTypingContext &context)
    -> std::optional<ConclusionOf<ast::Constant>> {
  if (!Super::checkConstant(constant, context))
    return std::nullopt;

  // Any integer constant is okay.
  if (!context.maxBitwidth)
    return ConclusionOf<ast::Constant>{};

  // Otherwise restrain it to our bitwidth.
  return std::visit(
      [&](auto &&value) -> ast::Constant {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_integral_v<T>) {
          // TODO: This is basically always a non-negative number.
          //       Figure out when/if it is safe to produce a negative number
          //       here!
          return {static_cast<T>(value & (1LL << *context.maxBitwidth) - 1)};
        }
        llvm_unreachable("double and float handled above");
      },
      constant.value);
}
