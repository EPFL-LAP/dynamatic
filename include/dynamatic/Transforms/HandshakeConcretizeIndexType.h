//===- HandshakeConcretizeIndexType.h - Index -> Integer --------*- C++ -*-===//
//
// This file declares the --handshake-concretize-index-type pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKECONCRETIZEINDEXTYPE_H
#define DYNAMATIC_TRANSFORMS_HANDSHAKECONCRETIZEINDEXTYPE_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {

#define GEN_PASS_DECL_HANDSHAKECONCRETIZEINDEXTYPE
#define GEN_PASS_DEF_HANDSHAKECONCRETIZEINDEXTYPE
#include "dynamatic/Transforms/Passes.h.inc"

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHandshakeConcretizeIndexType(unsigned width = 64);

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKECONCRETIZEINDEXTYPE_H