#ifndef BANBA_ATTRS_TD
#define BANBA_ATTRS_TD

include "mlir/IR/AttrTypeBase.td"

// Stores the LSQ's parameters inside the IR itself. Declaring the fields here
// gives each a named, typed getter, which a generic dictionary would not
def Banba_ConfigAttr : AttrDef<Banba_Dialect, "Config"> {
  // Name of the attribute in the printed IR
  let mnemonic = "config";

  let parameters = (ins
      // Identity
      StringRefParameter<>:$name,
      "bool":$master,
      // Widths
      "unsigned":$dataWidth,
      "unsigned":$addrWidth,
      "unsigned":$indexWidth,
      // Depths
      "unsigned":$fifoDepth,
      "unsigned":$fifoDepth_L,
      "unsigned":$fifoDepth_S,
      // Counts
      "unsigned":$numBBs,
      "unsigned":$numLoadPorts,
      "unsigned":$numStorePorts,
      "unsigned":$numLdChannels,
      "unsigned":$numStChannels,
      "DenseI32ArrayAttr":$numLoads,
      "DenseI32ArrayAttr":$numStores,
      // Order of the accesses within each group
      ArrayRefParameter<"DenseI32ArrayAttr">:$loadOffsets,
      ArrayRefParameter<"DenseI32ArrayAttr">:$storeOffsets,
      ArrayRefParameter<"DenseI32ArrayAttr">:$loadPorts,
      ArrayRefParameter<"DenseI32ArrayAttr">:$storePorts,
      ArrayRefParameter<"DenseI32ArrayAttr">:$ldOrder,
      ArrayRefParameter<"DenseI32ArrayAttr">:$ldPortIdx,
      ArrayRefParameter<"DenseI32ArrayAttr">:$stPortIdx);

  // Print each value together with its name
  let assemblyFormat = "`<` struct(params) `>`";

  // Declares a way of creating the attribute from an LsqConfig struct,
  // which is defined in BanbaDialect.cpp
  let builders = [AttrBuilder<(ins "const ::LsqConfig &":$config)>];
}

#endif // BANBA_ATTRS_TD
