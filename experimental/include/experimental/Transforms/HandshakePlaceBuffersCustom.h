//===- HandshakePlaceBuffersCustom.h - Place buffers in DFG -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-place-buffers-custom pass
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_TRANSFORMS_BUFFERPLACEMENT_PLACEBUFFERS_CUSTOM_H
#define EXPERIMENTAL_TRANSFORMS_BUFFERPLACEMENT_PLACEBUFFERS_CUSTOM_H

#include "dynamatic/Support/DynamaticPass.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace dynamatic {
namespace experimental {

// import auto-generate base class declaration
// options struct
// and create function
#define GEN_PASS_DECL_HANDSHAKEPLACEBUFFERSCUSTOM
#include "experimental/Transforms/Passes.h.inc"

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_TRANSFORMS_BUFFERPLACEMENT_PLACEBUFFERS_CUSTOM_H
