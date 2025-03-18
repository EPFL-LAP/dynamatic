//===----------- HandshakeInsertSkippableseq.h -------------------*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-insert-skippable-seq
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKEINSERTSKIPPABLESEQ_H
#define DYNAMATIC_TRANSFORMS_HANDSHAKEINSERTSKIPPABLESEQ_H

#include "dynamatic/Support/DynamaticPass.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace dynamatic {

#define GEN_PASS_DECL_HANDSHAKEINSERTSKIPPABLESEQ
#define GEN_PASS_DEF_HANDSHAKEINSERTSKIPPABLESEQ
#include "dynamatic/Transforms/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass>
createHandshakeInsertSkippableSeq(const unsigned N = 3);

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKEINSERTSKIPPABLESEQ_H
