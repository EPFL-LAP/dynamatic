#ifndef DYNAMATIC_TOOLS_BACKEND_LSQ_GENERATOR_BANBA_LOWERTOHW_H
#define DYNAMATIC_TOOLS_BACKEND_LSQ_GENERATOR_BANBA_LOWERTOHW_H

#include "mlir/IR/BuiltinOps.h"

namespace banba {

/// Lowers the banba dialect to hw modules and instances:
/// every banba op in the module becomes a hw.module
/// that holds the contents of its region,
/// and is replaced by a hw.instance of that module
void lowerToHw(mlir::ModuleOp modOp);

} // namespace banba

#endif // DYNAMATIC_TOOLS_BACKEND_LSQ_GENERATOR_BANBA_LOWERTOHW_H
