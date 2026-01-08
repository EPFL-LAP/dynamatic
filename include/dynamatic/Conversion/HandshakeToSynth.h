//===- HandshakeToSynth.h - Convert Handshake to Synth ----------------*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --lower-handshake-to-synth conversion pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_CONVERSION_HANDSHAKETOSYNTH_H
#define DYNAMATIC_CONVERSION_HANDSHAKETOSYNTH_H

#include "dynamatic/Support/DynamaticPass.h"

namespace dynamatic {

namespace synth {
/// Forward declare the Synth dialect which the pass depends on.
class SynthDialect;
} // namespace synth

namespace hw {
/// Forward declare the HW dialect which the pass depends on.
class HWDialect;
} // namespace hw

// Keywords for data and control signals when unbundling Handshake types using
// enums.
enum SignalKind {
  DATA_SIGNAL = 0,
  VALID_SIGNAL = 1,
  READY_SIGNAL = 2,
};

#define GEN_PASS_DECL_HANDSHAKETOSYNTH
#define GEN_PASS_DEF_HANDSHAKETOSYNTH
#include "dynamatic/Conversion/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass> createHandshakeToSynthPass();

} // namespace dynamatic

#endif // DYNAMATIC_CONVERSION_HANDSHAKETOSYNTH_H