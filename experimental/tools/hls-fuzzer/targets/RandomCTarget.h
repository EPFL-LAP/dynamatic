#ifndef DYNAMATIC_HLS_FUZZER_TARGETS_RANDOMCTARGET
#define DYNAMATIC_HLS_FUZZER_TARGETS_RANDOMCTARGET

#include "../AbstractTarget.h"

namespace dynamatic {

/// Target that generates unbiased random C programs. This restricts the base
/// generator as little as possible, only excluding some expressions that
/// dynamatic is known not to be able to compile.
class RandomCTarget : public AbstractTarget {
public:
  std::unique_ptr<AbstractWorker>
  createGenerator(const Options &options, Randomly randomly) const override;
};

} // namespace dynamatic

#endif
