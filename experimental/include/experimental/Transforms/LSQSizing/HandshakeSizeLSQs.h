//===- HandshakeSizeLSQs.h - Sizes the LSQs --------*- C++ -*-===//
//
// This file declares the --handshake-size-lsqs pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_SIZE_LSQS_H
#define DYNAMATIC_TRANSFORMS_SIZE_LSQS_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/TimingModels.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {
namespace experimental {
namespace lsqsizing {

#define GEN_PASS_DECL_HANDSHAKESIZELSQS
#define GEN_PASS_DEF_HANDSHAKESIZELSQS
#include "experimental/Transforms/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass>
createHandshakeSizeLSQs(StringRef timingModels = "", StringRef collisions = "");

} // namespace lsqsizing
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_SIZE_LSQS_H