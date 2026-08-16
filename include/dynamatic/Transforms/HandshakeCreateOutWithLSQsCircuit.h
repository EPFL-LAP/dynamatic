//===----------- HandshakeCreateOutWithLSQsCircuit.h -------------------*- C++
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

#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKECREATEOUTWITHLSQSCIRCUIT_H
#define DYNAMATIC_TRANSFORMS_HANDSHAKECREATEOUTWITHLSQSCIRCUIT_H

namespace dynamatic {

constexpr llvm::StringLiteral SKIP_COND_GEN("Skip.Condition_Generator");
constexpr llvm::StringLiteral SKIP_COND_SEQ("Skip.Conditional_Sequentializer");

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKECREATEOUTWITHLSQSCIRCUIT_H
