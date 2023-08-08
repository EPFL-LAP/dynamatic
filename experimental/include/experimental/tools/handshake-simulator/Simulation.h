//===- Simulation.h - Handshake MLIR Operations ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains functions used to execute a restricted form of the
// standard dialect, and the handshake dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HANDSHAKE_SIMULATION_H
#define CIRCT_DIALECT_HANDSHAKE_SIMULATION_H

#include "circt/Support/JSON.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringMap.h"
#include <string>

namespace dynamatic {
namespace experimental {

bool simulate(llvm::StringRef toplevelFunction,
              llvm::ArrayRef<std::string> inputArgs,
              mlir::OwningOpRef<mlir::ModuleOp> &module,
              mlir::MLIRContext &context,
              llvm::StringMap<std::string> &funcMap);
} // namespace dynamatic
} // namespace experimental


#endif
