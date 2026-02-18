//===- HandshakeChooseLSQType.h - Choosing LSQ Type -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the a pass to specify which type of LSQ to use
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_TRANSFORMS_CHOOSE_LSQ_TYPE_H
#define EXPERIMENTAL_TRANSFORMS_CHOOSE_LSQ_TYPE_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"

namespace dynamatic {
namespace experimental {

#define GEN_PASS_DECL_HANDSHAKECHOOSELSQTYPE
#include "experimental/Transforms/Passes.h.inc"

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_TRANSFORMS_CHOOSE_LSQ_TYPE_H
