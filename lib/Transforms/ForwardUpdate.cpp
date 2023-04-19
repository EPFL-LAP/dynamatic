//===- ForwardUpdate.cpp ---------*- C++ -*-===//
//
// This file declares functions for forward pass in --optimize-bits.
//
//===----------------------------------------------------------------------===//
#include "dynamatic/Transforms/ForwardUpdate.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"


