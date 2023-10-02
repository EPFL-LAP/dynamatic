//===- HandshakeFixArgNames.h - Match argument names with C --00-*- C++ -*-===//
//
// This file declares the --handshake-fix-arg-names pass.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_TRANSFORMS_HANDHSHAKEFIXARGNAMES_H
#define EXPERIMENTAL_TRANSFORMS_HANDHSHAKEFIXARGNAMES_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {
namespace experimental {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHandshakeFixArgNames(const std::string &source = "");

#define GEN_PASS_DECL_HANDSHAKEFIXARGNAMES
#define GEN_PASS_DEF_HANDSHAKEFIXARGNAMES
#include "experimental/Transforms/Passes.h.inc"

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_TRANSFORMS_HANDHSHAKEFIXARGNAMES_H
