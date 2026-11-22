#ifndef BANBA_MERGE_IDENTICAL_MODULES_H
#define BANBA_MERGE_IDENTICAL_MODULES_H

#include "mlir/IR/BuiltinOps.h"

namespace banba {

/// Merge the hw.modules that are the same hardware:
/// the first of each is kept, and the instances of the others point at it
void mergeIdenticalModules(mlir::ModuleOp modOp);

} // namespace banba

#endif // BANBA_MERGE_IDENTICAL_MODULES_H
