//===- FlattenMemRefRowMajor.h - Flatten memory accesses --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --flatten-memref-row-major pass, which is almost
// identical to CIRCT's --flatten-memref pass but uses row-major indexing for
// converting multidimensional load and store operations.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_FLATTENMEMREFROWMAJOR_H
#define DYNAMATIC_TRANSFORMS_FLATTENMEMREFROWMAJOR_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"

namespace dynamatic {

std::unique_ptr<dynamatic::DynamaticPass> createFlattenMemRefRowMajorPass();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_FLATTENMEMREFROWMAJOR_H
