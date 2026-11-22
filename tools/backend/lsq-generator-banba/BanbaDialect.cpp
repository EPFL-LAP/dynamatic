#include "BanbaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace banba;

// The definition of the dialect, generated from Banba.td
#include "BanbaDialect.cpp.inc"

// The definitions of the dialect's attributes, generated from BanbaAttrs.td
#define GET_ATTRDEF_CLASSES
#include "BanbaAttrs.cpp.inc"

// The definitions of the dialect's interfaces,
// generated from BanbaInterfaces.td
#include "BanbaInterfaces.cpp.inc"

// The definitions of the dialect's operations,
// generated from the Ops.td file of each folder
#define GET_OP_CLASSES
#include "Banba.cpp.inc"

void BanbaDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Banba.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "BanbaAttrs.cpp.inc"
      >();
}

/// Converts a list of numbers into MLIR's integer-array attribute.
static DenseI32ArrayAttr toArrayAttr(Builder &builder,
                                     const std::vector<unsigned> &values) {
  SmallVector<int32_t> converted(values.begin(), values.end());
  return builder.getDenseI32ArrayAttr(converted);
}

/// Converts a list of lists of numbers into a list of MLIR's integer-array
/// attributes.
static SmallVector<DenseI32ArrayAttr>
toArrayAttrs(Builder &builder, const std::vector<std::vector<unsigned>> &rows) {
  SmallVector<DenseI32ArrayAttr> converted;
  for (const std::vector<unsigned> &row : rows)
    converted.push_back(toArrayAttr(builder, row));
  return converted;
}

ConfigAttr ConfigAttr::get(MLIRContext *context, const LsqConfig &config) {
  Builder builder(context);
  // The arguments below are matched to the attribute's fields by position:
  // they must stay in the same order as the fields in BanbaAttrs.td
  // clang-format off
  return ConfigAttr::get(context,
      // Identity
      config.name,
      config.master,
      // Widths
      config.dataWidth,
      config.addrWidth,
      config.indexWidth,
      // Depths
      config.fifoDepth,
      config.fifoDepth_L,
      config.fifoDepth_S,
      // Counts
      config.numBBs,
      config.numLoadPorts,
      config.numStorePorts,
      config.numLdChannels,
      config.numStChannels,
      toArrayAttr(builder, config.numLoads),
      toArrayAttr(builder, config.numStores),
      // Order of the accesses within each group
      toArrayAttrs(builder, config.loadOffsets),
      toArrayAttrs(builder, config.storeOffsets),
      toArrayAttrs(builder, config.loadPorts),
      toArrayAttrs(builder, config.storePorts),
      toArrayAttrs(builder, config.ldOrder),
      toArrayAttrs(builder, config.ldPortIdx),
      toArrayAttrs(builder, config.stPortIdx));
  // clang-format on
}
