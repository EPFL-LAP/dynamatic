#ifndef BANBA_REDUCE_CASTS_TO_INTEGERS_H
#define BANBA_REDUCE_CASTS_TO_INTEGERS_H

#include "mlir/IR/BuiltinOps.h"

namespace banba {

/// Reduce the casts between arrays of 1-bit values and integers
/// in a design that is lowered to hw modules
void reduceCastsToIntegers(mlir::ModuleOp modOp);

} // namespace banba

#endif // BANBA_REDUCE_CASTS_TO_INTEGERS_H
