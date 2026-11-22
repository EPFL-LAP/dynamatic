#ifndef BANBA_TD
#define BANBA_TD

include "mlir/IR/OpBase.td"

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

#endif // BANBA_TD
