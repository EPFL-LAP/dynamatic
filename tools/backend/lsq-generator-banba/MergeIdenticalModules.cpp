#include "MergeIdenticalModules.h"
#include "BanbaOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;
using namespace banba;

/// Whether two hw.modules are the same hardware:
/// lowered from the same operation, with the same attributes,
/// and with the same ports.
/// Only modules that are written to the same output file count
static bool areIdentical(circt::hw::HWModuleOp a, circt::hw::HWModuleOp b) {
  bool sameOp = a->getAttr(opNameAttrName) == b->getAttr(opNameAttrName);
  bool sameAttrs = a->getAttr(opAttrsAttrName) == b->getAttr(opAttrsAttrName);
  bool samePorts = a.getModuleType() == b.getModuleType();
  bool sameFile = a->getAttr("output_file") == b->getAttr("output_file");
  return sameOp && sameAttrs && samePorts && sameFile;
}

/// Merge the hw.modules that are the same hardware:
/// the first of each is kept, and the instances of the others point at it
void banba::mergeIdenticalModules(ModuleOp modOp) {
  // The modules that are kept
  SmallVector<circt::hw::HWModuleOp> kept;

  // The modules that are removed
  SmallVector<circt::hw::HWModuleOp> removed;

  // Map of the name of a removed module -> the kept module that replaces it
  llvm::StringMap<circt::hw::HWModuleOp> replacementOf;

  // For each module in the design
  for (auto hwMod : modOp.getOps<circt::hw::HWModuleOp>()) {
    // Look for a kept module that it is identical to
    auto *identical = llvm::find_if(kept, [&](circt::hw::HWModuleOp other) {
      return areIdentical(other, hwMod);
    });

    if (identical == kept.end()) {
      // There is none, so the module is kept
      kept.push_back(hwMod);
    } else {
      // There is one, so the module is removed, and that one replaces it
      removed.push_back(hwMod);
      replacementOf[hwMod.getName()] = *identical;
    }
  }

  // For each instance inside each kept module
  for (circt::hw::HWModuleOp parent : kept) {
    for (auto instance :
         parent.getBodyBlock()->getOps<circt::hw::InstanceOp>()) {
      // If it is an instance of a removed module,
      // point it at the kept module that replaces that one
      auto replacement = replacementOf.find(instance.getReferencedModuleName());
      if (replacement != replacementOf.end())
        instance.setModuleName(replacement->second.getName());
    }
  }

  // Remove the comment of each removed module.
  // The lowering recorded on each comment the name of its module
  for (Operation &op : llvm::make_early_inc_range(*modOp.getBody())) {
    auto commentOf = op.getAttrOfType<StringAttr>(commentOfAttrName);
    if (commentOf && replacementOf.count(commentOf.getValue()))
      op.erase();
  }

  // Remove the modules themselves
  for (circt::hw::HWModuleOp hwMod : removed)
    hwMod.erase();
}
