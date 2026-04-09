#ifndef DYNAMATIC_HLS_FUZZER_TARGETS_LSQOPTIMIZATIONSTARGET
#define DYNAMATIC_HLS_FUZZER_TARGETS_LSQOPTIMIZATIONSTARGET

#include "hls-fuzzer/AbstractTarget.h"

namespace dynamatic {

class LSQOptimizationsTarget : public AbstractTarget {
public:
  std::unique_ptr<AbstractWorker>
  createWorker(const Options &options, Randomly randomly) const override;
};

} // namespace dynamatic

#endif
