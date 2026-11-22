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

  // Values derived from the fields above
  let extraClassDeclaration = [{
    /// The width of a pointer into the load queue
    unsigned getLdPointerWidth() const {
      return Log2_32_Ceil(getFifoDepth_L());
    }

    /// The width of a pointer into the store queue
    unsigned getStPointerWidth() const {
      return Log2_32_Ceil(getFifoDepth_S());
    }

    /// The width of a number of load queue entries,
    /// from zero up to all of them
    unsigned getLdCountWidth() const {
      return Log2_32_Ceil(getFifoDepth_L() + 1);
    }

    /// The width of a number of store queue entries,
    /// from zero up to all of them
    unsigned getStCountWidth() const {
      return Log2_32_Ceil(getFifoDepth_S() + 1);
    }

    /// The width of the index of a load port.
    /// One bit even when there is a single load port
    unsigned getLdPortIdxWidth() const {
      return std::max(1u, Log2_32_Ceil(getNumLoadPorts()));
    }

    /// The width of the index of a store port.
    /// One bit even when there is a single store port
    unsigned getStPortIdxWidth() const {
      return std::max(1u, Log2_32_Ceil(getNumStorePorts()));
    }

    /// The width of the number of free entries in the load queue:
    /// one bit more than a pointer into it
    unsigned getLdFreeEntriesWidth() const { return getLdPointerWidth() + 1; }

    /// The width of the number of free entries in the store queue:
    /// one bit more than a pointer into it
    unsigned getStFreeEntriesWidth() const { return getStPointerWidth() + 1; }
  }];
}

#endif // BANBA_ATTRS_TD
