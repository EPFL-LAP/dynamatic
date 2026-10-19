#ifndef DYNAMATIC_DIALECT_CFEXTRA_CFEXTRAOPS_H
#define DYNAMATIC_DIALECT_CFEXTRA_CFEXTRAOPS_H

#include "dynamatic/Dialect/CFExtra/CFExtraDialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "dynamatic/Dialect/CFExtra/CFExtra.h.inc"

#endif // DYNAMATIC_DIALECT_CFEXTRA_CFEXTRAOPS_H