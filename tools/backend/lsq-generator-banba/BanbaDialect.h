#ifndef DYNAMATIC_TOOLS_BACKEND_LSQ_GENERATOR_BANBA_BANBADIALECT_H
#define DYNAMATIC_TOOLS_BACKEND_LSQ_GENERATOR_BANBA_BANBADIALECT_H

#include "LsqConfig.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"

// The declaration of the dialect, generated from Banba.td
#include "BanbaDialect.h.inc"

namespace banba {
// The names from MLIR and LLVM that the dialect's code,
// including the generated code, uses without writing their namespace
using mlir::DenseI32ArrayAttr;
using llvm::cast;
using llvm::isa;
} // namespace banba

// The declarations of the dialect's attributes, generated from BanbaAttrs.td
#define GET_ATTRDEF_CLASSES
#include "BanbaAttrs.h.inc"

#endif // DYNAMATIC_TOOLS_BACKEND_LSQ_GENERATOR_BANBA_BANBADIALECT_H
