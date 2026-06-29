#ifndef DYNAMATIC_HLS_FUZZER_TARGET_RANDOMCTYPESYSTEM
#define DYNAMATIC_HLS_FUZZER_TARGET_RANDOMCTYPESYSTEM

#include "BitwidthTypeSystem.h"
#include "DynamaticTypeSystem.h"
#include "hls-fuzzer/ConjunctionTypeSystem.h"
#include "hls-fuzzer/LimitTypeSystem.h"
#include "hls-fuzzer/OptionalTypeSystem.h"
#include "hls-fuzzer/TypeSystem.h"

namespace dynamatic::gen {
/// Type system for the '--random-c' target.
/// Uses a combination of the dynamatic type system and the limit type system.
/// For loop bounds, the bitwidth type system is selectively enabled to generate
/// reasonable loop bounds.
class RandomCTypeSystem final
    : public ConjunctionTypeSystemBase<RandomCTypeSystem, DynamaticTypeSystem,
                                       LimitTypeSystem,
                                       OptionalTypeSystem<BitwidthTypeSystem>> {
public:
  explicit RandomCTypeSystem(Randomly &random)
      : ConjunctionTypeSystemBase(DynamaticTypeSystem(), LimitTypeSystem(),
                                  // Upper bound on what bitwidth can be
                                  // generated in unrestricted contexts.
                                  BitwidthTypeSystem(64, random)) {}

  TransferFnArray<ast::StructuredForStatement>
  getStructuredForStatementTransferFns() override;
};
} // namespace dynamatic::gen
#endif
