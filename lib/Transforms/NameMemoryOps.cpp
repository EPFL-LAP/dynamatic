//===- NameMemoryOps.cpp - Give a unique name to all memory ops -*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --name-memory-ops pass, which sets a MemAccessNameAttr
// attribute containing a unique access name on all supported memory operations
// in a module (names are unique within the context of each function). This pass
// is a prerequisite for running the memory analysis pass, which uses these
// attributes to allow memory accesses to reference one another.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/NameMemoryOps.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace dynamatic;
using namespace circt;
using namespace circt::handshake;

namespace {
/// Helper struct to create unique names for all memory accesses refering to a
/// specific memref.
struct AccessNameUniquer {
  /// Creates a AccessNameUniquer using a default prefix name.
  AccessNameUniquer() : name("default") {}

  /// Creates a AccessNameUniquer using a specific prefix name.
  AccessNameUniquer(std::string name) : name(name) {}

  /// Returns a unique name for the operation passed as argument. If the same
  /// operation is passed on multiple calls, it will receive a different unique
  /// name each time.
  std::string addOp(Operation *op) {
    if (isa<affine::AffineLoadOp, memref::LoadOp>(op))
      return name + "_load" + std::to_string(numLoads++);
    return name + "_store" + std::to_string(numStores++);
  }

private:
  /// The prefix name for all accesses.
  std::string name;
  /// The number of load accesses already encountered.
  unsigned numLoads = 0;
  /// The number of store accesses already encountered.
  unsigned numStores = 0;
};
} // namespace

/// Extracts the memref of supported memory operations. When the given operation
/// is a supported memory operation, puts the extracted memref value in `out`
/// and succeeds; fails otherwise.
static LogicalResult getMemrefOfMemoryOp(Operation *op, Value &out) {
  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case<memref::LoadOp, memref::StoreOp>([&](auto memrefOp) {
        out = memrefOp.getMemRef();
        return success();
      })
      .Case<affine::AffineLoadOp, affine::AffineStoreOp>([&](auto) {
        affine::MemRefAccess access(op);
        out = access.memref;
        return success();
      })
      .Default([&](auto) { return failure(); });
}

/// Sets a MemAccessNameAttr attribute containing a unique name for each memory
/// access in the function. Fails if one memory access already has one such
/// attribute.
static LogicalResult nameMemoryOps(func::FuncOp funcOp, MLIRContext *ctx) {

  DenseMap<Value, AccessNameUniquer> memrefNames;
  // Generate a unique name for each memory operation in the function
  auto accessNameUniquer = [&](Value memref, Operation *memOp) {
    std::string baseName;
    if (!memrefNames.contains(memref))
      memrefNames.insert(std::make_pair(
          memref,
          AccessNameUniquer("mem" + std::to_string(memrefNames.size()))));
    return memrefNames[memref].addOp(memOp);
  };

  // Set an attribute on each memory operation to hold a unique name for it
  auto attrName = MemAccessNameAttr::getMnemonic();
  LogicalResult allMemOpsValid = success();
  funcOp->walk([&](Operation *op) {
    Value memref;
    if (failed(getMemrefOfMemoryOp(op, memref)))
      return;
    if (op->getAttrOfType<MemAccessNameAttr>(attrName)) {
      // Emit an error if the attribute already exists on the operation and fail
      // the pass later
      allMemOpsValid = op->emitError()
                       << "op already has an '" << attrName << "' attribute";
      return;
    }
    op->setAttr(attrName,
                MemAccessNameAttr::get(ctx, accessNameUniquer(memref, op)));
  });

  if (failed(allMemOpsValid))
    return funcOp->emitError()
           << "some memory ops in the function already had an '"
           << MemAccessNameAttr::getMnemonic()
           << "' attribute before the pass ran, which is illegal";

  return success();
}

namespace {

/// Simple driver for memory ops naming pass. Runs the pass on every function in
/// the module independently and succeeds whenever the transformation succeeded
/// for every function.
struct NameMemoryOpsPass : public NameMemoryOpsBase<NameMemoryOpsPass> {

  void runOnOperation() override {
    for (auto funcOp : getOperation().getOps<func::FuncOp>())
      if (failed(nameMemoryOps(funcOp, &getContext())))
        return signalPassFailure();
  };
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createNameMemoryOps() {
  return std::make_unique<NameMemoryOpsPass>();
}
