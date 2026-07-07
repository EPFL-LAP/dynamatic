#include "RandomCTypeSystem.h"

dynamatic::gen::TransferFnArray<dynamatic::ast::StructuredForStatement>
dynamatic::gen::RandomCTypeSystem::getStructuredForStatementTransferFns() {
  return combineGetTransferFns<ast::StructuredForStatement>(
      [&](auto &&typeSystem) {
        if constexpr (std::is_same_v<std::decay_t<decltype(typeSystem)>,
                                     OptionalTypeSystem<BitwidthTypeSystem>>) {
          // Allow loops to run at most 8 times.
          return OptionalTypeSystem<BitwidthTypeSystem>::enableTypeSystemFor<
              ast::StructuredForStatement, ast::StructuredForStatement::START,
              ast::StructuredForStatement::END,
              ast::StructuredForStatement::STEP>(
              typeSystem.getStructuredForStatementTransferFns(),
              BitwidthTypingContext(3));
        } else {
          return typeSystem.getStructuredForStatementTransferFns();
        }
      });
}
