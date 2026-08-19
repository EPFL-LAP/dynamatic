#include "TerminationTypeSystem.h"

dynamatic::gen::TransferFnArray<dynamatic::ast::StructuredForStatement>
dynamatic::gen::TerminationTypeSystem::getStructuredForStatementTransferFns() {
  return combineGetTransferFns([&](auto &&typeSystem) {
    if constexpr (std::is_same_v<std::decay_t<decltype(typeSystem)>,
                                 OptionalTypeSystem<BitwidthTypeSystem>>) {
      // Enable the bitwidth type system for the loop bounds only. This
      // bounds the values 'start', 'end' and 'step' can take and therefore
      // the number of times the loop runs (at most 8 times here).
      return OptionalTypeSystem<BitwidthTypeSystem>::enableTypeSystemFor<
          ast::StructuredForStatement::START, ast::StructuredForStatement::END,
          ast::StructuredForStatement::STEP>(
          typeSystem.getStructuredForStatementTransferFns(),
          BitwidthTypingContext(3));
    } else {
      return typeSystem.getStructuredForStatementTransferFns();
    }
  });
}
