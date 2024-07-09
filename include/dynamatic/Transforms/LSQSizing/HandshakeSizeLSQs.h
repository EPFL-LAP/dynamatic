//===- HandshakeSizeLSQs.h - Sizes the LSQs --------*- C++ -*-===//
//
// This file declares the --handshake-size-lsqs pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_SIZE_LSQS_H
#define DYNAMATIC_TRANSFORMS_SIZE_LSQS_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {

#define GEN_PASS_DECL_HANDSHAKESIZELSQS
#define GEN_PASS_DEF_HANDSHAKESIZELSQS
#include "dynamatic/Transforms/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass>
createHandshakeSizeLSQs();

} // namespace dynamatic


#endif // DYNAMATIC_TRANSFORMS_SIZE_LSQS_H