//===- PassDetails.h - Handshake passes class details -----------*- C++ -*-===//
//
// This is the header file for all handshake passes defined in Dynamatic. It
// contains forward declarations needed by handshake passes and includes
// auto-generated base class definitions for all handshake passes.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_DIALECT_HANDSHAKE_PASSDETAILS_H
#define DYNAMATIC_DIALECT_HANDSHAKE_PASSDETAILS_H

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {

#define GEN_PASS_CLASSES
#include "dynamatic/Dialect/Handshake/HandshakePasses.h.inc"

} // namespace dynamatic

#endif // DYNAMATIC_DIALECT_HANDSHAKE_PASSDETAILS_H
