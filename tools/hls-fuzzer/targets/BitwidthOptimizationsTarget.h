#ifndef DYNAMATIC_HLS_FUZZER_TARGETS_BITWIDTHOPTIMIZATIONSTARGET
#define DYNAMATIC_HLS_FUZZER_TARGETS_BITWIDTHOPTIMIZATIONSTARGET

#include "hls-fuzzer/AbstractTarget.h"

namespace dynamatic {

class BitwidthOptimizationsTarget : public AbstractTarget {
public:
  std::unique_ptr<AbstractWorker>
  createWorker(const Options &options, Randomly randomly) const override;
};

} // namespace dynamatic

#endif
