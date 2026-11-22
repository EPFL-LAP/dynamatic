#ifndef BANBA_TD
#define BANBA_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Banba_Dialect : Dialect {
  let name = "banba";
  let cppNamespace = "::banba";

  let summary = "Parameterized load-store-queue hardware described as IR";
  let description = [{
    The `banba` dialect represents a Dynamatic load-store queue as custom
    operations containing sub-regions.
  }];

  let useDefaultAttributePrinterParser = 1;
}

include "BanbaAttrs.td"
include "BanbaInterfaces.td"
include "common/CommonOps.td"
include "common/CallerParamOps.td"
include "shared-queues/SharedQueueOps.td"
include "per-port-queues/PerPortQueueOps.td"

#endif // BANBA_TD
