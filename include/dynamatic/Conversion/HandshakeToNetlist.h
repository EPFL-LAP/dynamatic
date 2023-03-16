//===- HandshakeToNetlist.h - Converts handshake to HW/ESI ------*- C++ -*-===//
//
// This file declares the --lower-handshake-to-netlist conversion pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_CONVERSION_HANDSHAKETONETLIST_H
#define DYNAMATIC_CONVERSION_HANDSHAKETONETLIST_H

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace circt::hw;
using namespace circt::esi;

namespace dynamatic {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHandshakeToNetlistPass();

} // namespace dynamatic

#endif // DYNAMATIC_CONVERSION_HANDSHAKETONETLIST_H
