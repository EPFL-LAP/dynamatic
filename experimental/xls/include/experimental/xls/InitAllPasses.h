//===- InitAllPasses.h - XLS passes registration -----------------*- C++-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all XLS passes
// defined in the Dynamatic.
//
//===----------------------------------------------------------------------===//

#ifndef XLS_INITALLPASSES_H
#define XLS_INITALLPASSES_H

#include "experimental/xls/Conversion/Passes.h"

namespace dynamatic {
namespace experimental {
namespace xls {

inline void registerAllPasses() {
  experimental::xls::registerConversionPasses();
}

} // namespace xls
} // namespace experimental
} // namespace dynamatic

#endif // XLS_INITALLPASSES_H
