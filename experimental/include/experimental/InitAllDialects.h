//===- InitAllDialects.h - Dynamatic dialects registration -------*- C++-*-===//
//
// This file defines a helper to trigger the registration of all dialects that
// Dynamatic users may care about.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_INITALLDIALECTS_H
#define EXPERIMENTAL_INITALLDIALECTS_H

#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dialect.h"

namespace dynamatic {

inline void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::LLVM::LLVMDialect, mlir::affine::AffineDialect,
                  mlir::memref::MemRefDialect, mlir::func::FuncDialect,
                  mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                  mlir::scf::SCFDialect, circt::handshake::HandshakeDialect>();
}

} // namespace dynamatic

#endif // EXPERIMENTAL_INITALLDIALECTS_H
