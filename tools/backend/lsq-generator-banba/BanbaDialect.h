#ifndef DYNAMATIC_TOOLS_BACKEND_LSQ_GENERATOR_BANBA_BANBADIALECT_H
#define DYNAMATIC_TOOLS_BACKEND_LSQ_GENERATOR_BANBA_BANBADIALECT_H

#include "LsqConfig.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

// The declaration of the dialect, generated from Banba.td
#include "BanbaDialect.h.inc"

namespace banba {
// The names from MLIR, LLVM and CIRCT that the dialect's code,
// including the generated code, uses without writing their namespace
using circt::seq::ClockType;
using llvm::cast;
using llvm::isa;
using llvm::SmallVector;
using llvm::StringLiteral;
using mlir::DenseI32ArrayAttr;
using mlir::IntegerType;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::Operation;
using mlir::success;
using mlir::Type;
using mlir::TypeRange;
using mlir::OpTrait::TraitBase;
} // namespace banba

// The declarations of the dialect's attributes, generated from BanbaAttrs.td
#define GET_ATTRDEF_CLASSES
#include "BanbaAttrs.h.inc"

#endif // DYNAMATIC_TOOLS_BACKEND_LSQ_GENERATOR_BANBA_BANBADIALECT_H
