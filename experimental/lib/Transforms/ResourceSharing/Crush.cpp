//===- FCCM22Sharing.cpp - Resource Sharing ---------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/ResourceSharing/Crush.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "dynamatic/Transforms/BufferPlacement/HandshakePlaceBuffers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <map>
#include <string>

using namespace llvm;
using namespace dynamatic;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::sharing;

using namespace dynamatic::buffer;

// extracted data from buffer placement
struct PerfOptInfo {

  // For each CFDFC, the obtained throughput
  std::map<CFDFC, double> cfdfcThroughput;
};

namespace {

// An wrapper class that applies buffer p
// extracts the report.
struct BufferPlacementWrapperPass : public HandshakePlaceBuffersPass {
  BufferPlacementWrapperPass(PerfOptInfo &info, StringRef algorithm,
                             StringRef frequencies, StringRef timingModels,
                             bool firstCFDFC, double targetCP, unsigned timeout,
                             bool dumpLogs)
      : HandshakePlaceBuffersPass(algorithm, frequencies, timingModels,
                                  firstCFDFC, targetCP, timeout, dumpLogs),
        info(info){};
  PerfOptInfo &info;

protected:
  void runDynamaticPass() override {
    HandshakePlaceBuffersPass::runDynamaticPass();
  }
};

struct CreditBasedSharingPass
    : public dynamatic::experimental::sharing::impl::CreditBasedSharingBase<
          CreditBasedSharingPass> {

  CreditBasedSharingPass(StringRef algorithm, StringRef frequencies,
                         StringRef timingModels, bool firstCFDFC,
                         double targetCP, unsigned timeout, bool dumpLogs) {
    this->algorithm = algorithm.str();
    this->frequencies = frequencies.str();
    this->timingModels = timingModels.str();
    this->firstCFDFC = firstCFDFC;
    this->targetCP = targetCP;
    this->timeout = timeout;
    this->dumpLogs = dumpLogs;
  }

  void runDynamaticPass() override;
};
} // namespace

void CreditBasedSharingPass::runDynamaticPass() {
  llvm::errs() << "***** Resource Sharing *****\n";

  ModuleOp modOp = getOperation();

  PerfOptInfo data;

  TimingDatabase timingDB(&getContext());
  if (failed(TimingDatabase::readFromJSON(timingModels, timingDB)))
    return signalPassFailure();

  // running buffer placement on current module
  mlir::PassManager pm(&getContext());
  pm.addPass(std::make_unique<BufferPlacementWrapperPass>(
      data, algorithm, frequencies, timingModels, firstCFDFC, targetCP, timeout,
      dumpLogs));
  if (failed(pm.run(modOp))) {
    return signalPassFailure();
  }
}

namespace dynamatic {
namespace experimental {
namespace sharing {

/// Returns a unique pointer to an operation pass that matches MLIR modules.
std::unique_ptr<dynamatic::DynamaticPass>
createCreditBasedSharing(StringRef algorithm, StringRef frequencies,
                         StringRef timingModels, bool firstCFDFC,
                         double targetCP, unsigned timeout, bool dumpLogs) {
  return std::make_unique<CreditBasedSharingPass>(algorithm, frequencies,
                                                  timingModels, firstCFDFC,
                                                  targetCP, timeout, dumpLogs);
}

} // namespace sharing
} // namespace experimental
} // namespace dynamatic
