//===- SwitchingEstimation.cpp - Estimate Swithicng Activities ------*- C++ -*-===//
//
// Implements the switching estimation pass for all untis in the generated
// dataflow circuit
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/Switching/SwitchingEstimation.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Support/Attribute.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <cmath>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::switching;

namespace {
// Define Switching Estimation pass driver
struct SwitchingEstimationPass
    : public dynamatic::experimental::switching::impl::SwitchingEstimationBase<
        SwitchingEstimationPass> {

  SwitchingEstimationPass(StringRef resultFolderPath,
                          StringRef timingModels) {
    this->resultFolderPath = resultFolderPath.str();
    this->timingModels = timingModels.str();
  }

  void runDynamaticPass() override;
};
} // namespace 

void SwitchingEstimationPass::runDynamaticPass() {
  // Read Component Latencies
  TimingDatabase timingDB(&getContext());
  if (failed(TimingDatabase::readFromJSON(timingModels, timingDB)))
    signalPassFailure();

}


namespace dynamatic {
namespace experimental {
namespace switching {

// Return a unique pointer for the switching estimation pass
std::unique_ptr<dynamatic::DynamaticPass>
createSwitchingEstimation(StringRef resultFolderPath, StringRef timingModels) {
  return std::make_unique<SwitchingEstimationPass>(resultFolderPath, timingModels);
}

} // namespace switching
} // namespace experimental
} // namespace dynamatic
