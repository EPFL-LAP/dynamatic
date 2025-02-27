//===- InitAllDialects.h - XLS dialects registration -------------*- C++-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all dialects that
// are required for the XLS integration.
//
//===----------------------------------------------------------------------===//

#ifndef XLS_INITALLDIALECTS_H
#define XLS_INITALLDIALECTS_H

#include "experimental/xls/Dialect/Xls/XlsDialect.h"

namespace dynamatic {
namespace experimental {
namespace xls {

inline void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::xls::XlsDialect>();
}

} // namespace xls
} // namespace experimental
} // namespace dynamatic

#endif // XLS_INITALLDIALECTS_H
