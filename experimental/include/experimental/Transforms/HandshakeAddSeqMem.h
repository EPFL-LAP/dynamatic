//===- HandshakeAnalyzeLSQUsage.h - LSQ flow analysis -----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-add-seq-mem pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKEADDSEQMEM_H
#define DYNAMATIC_TRANSFORMS_HANDSHAKEADDSEQMEM_H

#include "dynamatic/Support/DynamaticPass.h"

namespace dynamatic {
namespace experimental {

#define GEN_PASS_DECL_HANDSHAKEADDSEQMEM
#define GEN_PASS_DEF_HANDSHAKEADDSEQMEM
#include "experimental/Transforms/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass> createHandshakeAddSeqMem();

} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKEADDSEQMEM_H
