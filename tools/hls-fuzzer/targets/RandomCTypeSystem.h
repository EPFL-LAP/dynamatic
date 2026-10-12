#ifndef DYNAMATIC_HLS_FUZZER_TARGET_RANDOMCTYPESYSTEM
#define DYNAMATIC_HLS_FUZZER_TARGET_RANDOMCTYPESYSTEM

#include "DynamaticTypeSystem.h"
#include "TerminationTypeSystem.h"
#include "hls-fuzzer/ConjunctionTypeSystem.h"
#include "hls-fuzzer/LimitTypeSystem.h"
#include "hls-fuzzer/TypeSystem.h"

namespace dynamatic::gen {
/// Type system for the '--random-c' target.
/// Combines the dynamatic type system, the limit type system and the
/// termination type system. The latter ensures generated programs always
/// terminate by bounding loop counts and forbidding writes to loop iteration
/// variables.
class RandomCTypeSystem final
    : public ConjunctionTypeSystemBase<RandomCTypeSystem, DynamaticTypeSystem,
                                       LimitTypeSystem, TerminationTypeSystem> {
public:
  explicit RandomCTypeSystem(Randomly &random)
      : ConjunctionTypeSystemBase(DynamaticTypeSystem(),
                                  LimitTypeSystem(random),
                                  TerminationTypeSystem(random)) {}
};
} // namespace dynamatic::gen
#endif
