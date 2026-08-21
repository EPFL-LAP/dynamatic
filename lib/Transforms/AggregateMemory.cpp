//===- AggregateMemory.cpp - Aggregate Memory Pass ----------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the --hw-mark-li-memory-interface pass, which marks
// handshake memory interfaces as latency-insensitive.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/AggregateMemory.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Utils/Utils.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace dynamatic;

// Create a new aggregated memref argument for the function.
Value addAggregatedMemrefArg(func::FuncOp funcOp,
                             MemRefType aggregatedMemRefType,
                             int memrefNameId) {
  // Add the new argument to the entry block
  Block &entryBlock = funcOp.front();
  OpBuilder builder(funcOp.getContext());

  // Add the argument at the end of the argument list
  BlockArgument newArg =
      entryBlock.addArgument(aggregatedMemRefType, funcOp.getLoc());

  // Also update the function type
  auto funcType = funcOp.getFunctionType();
  SmallVector<Type> inputTypes(funcType.getInputs().begin(),
                               funcType.getInputs().end());
  SmallVector<Type> resultTypes(funcType.getResults().begin(),
                                funcType.getResults().end());

  inputTypes.push_back(aggregatedMemRefType);

  auto newFuncType = builder.getFunctionType(inputTypes, resultTypes);
  funcOp.setType(newFuncType);

  // Set the 'handshake.name' attribute to the new argument
  std::string memrefArgName = "aggregatedMemref" + std::to_string(memrefNameId);
  funcOp.setArgAttr(newArg.getArgNumber(), "handshake.arg_name",
                    builder.getStringAttr(memrefArgName));

  return newArg;
}

// Group memref arguments of a function by their element type
std::vector<std::pair<Type, std::vector<Value>>>
groupMemrefsByType(func::FuncOp funcOp) {
  std::vector<std::pair<Type, std::vector<Value>>> memrefByType;
  // Get the function's arguments that are memref types and group them by
  // type
  for (auto arg : funcOp.getArguments()) {
    if (arg.getType().isa<MemRefType>()) {
      MemRefType memRefType = arg.getType().cast<MemRefType>();
      Type elementType = memRefType.getElementType();
      // Check if we already have an entry for this type
      bool foundEntry = false;
      for (auto &memrefTypeAndValues : memrefByType) {
        if (memrefTypeAndValues.first == elementType) {
          memrefTypeAndValues.second.push_back(arg);
          foundEntry = true;
          break;
        }
      }
      if (!foundEntry) {
        memrefByType.push_back(
            std::make_pair(elementType, std::vector<Value>{arg}));
      }
    }
  }
  return memrefByType;
}

// Create a new aggregated memref argument for a function
Value createAggregegatedMemrefArg(func::FuncOp funcOp, Type elementType,
                                  std::vector<Value> &memrefValues,
                                  int memrefNameId) {
  // Create a new aggregated memref type that should be linear
  int totalSize = 0;
  for (auto memrefValue : memrefValues) {
    auto memRefType = memrefValue.getType().cast<MemRefType>();
    // Assert that the memref is statically sized and is 1D
    assert(memRefType.hasStaticShape() && "expected statically shaped memref");
    assert(memRefType.getShape().size() == 1 &&
           "expected unidimensional memref");
    // Accumulate total size
    totalSize += memRefType.getNumElements();
  }
  // Create the new aggregated memref type
  MemRefType aggregatedMemRefType = MemRefType::get({totalSize}, elementType);
  // Create a new argument for the aggregated memref
  OpBuilder builder(funcOp.getBody());
  auto aggregatedMemrefArg =
      addAggregatedMemrefArg(funcOp, aggregatedMemRefType, memrefNameId);
  return aggregatedMemrefArg;
}

// Replace the indeces of the original memref with the index into the aggregated
// memref
void replaceUsesOfMemrefIntoAggregatedMemref(func::FuncOp funcOp,
                                             Value originalMemref,
                                             Value aggregatedMemrefArg,
                                             int offset) {
  if (offset < 0) {
    llvm::report_fatal_error(
        "Offset for replacing memref uses into aggregated memref must be >= 0");
  }
  for (auto &use : llvm::make_early_inc_range(originalMemref.getUses())) {
    // Get previous index depending on the op type
    Value oldIndex;
    if (auto loadOp = dyn_cast<memref::LoadOp>(use.getOwner())) {
      oldIndex = loadOp.getIndices()[0];
    } else if (auto storeOp = dyn_cast<memref::StoreOp>(use.getOwner())) {
      oldIndex = storeOp.getIndices()[0];
    } else if (auto transferReadOp =
                   dyn_cast<vector::TransferReadOp>(use.getOwner())) {
      oldIndex = transferReadOp.getIndices()[0];
    } else {
      assert(false && "unsupported memref use operation");
    }
    // Create new index by adding currentOffset to oldIndex
    OpBuilder builder(use.getOwner());
    auto loc = use.getOwner()->getLoc();
    Value offsetValue;
    Value newIndex = oldIndex;
    // If offset == 0, no need to create a new index
    if (offset > 0) {
      offsetValue = builder.create<arith::ConstantOp>(
          loc, builder.getIndexType(), builder.getIndexAttr(offset));
      newIndex = builder.create<arith::AddIOp>(loc, builder.getIndexType(),
                                               oldIndex, offsetValue);
    }
    // Replace the use depending on the op type
    if (auto loadOp = dyn_cast<memref::LoadOp>(use.getOwner())) {
      // Operand 0 = memref, operand 1 = index
      loadOp.getOperation()->setOperand(0, aggregatedMemrefArg);
      loadOp.getOperation()->setOperand(1, newIndex);
    } else if (auto storeOp = dyn_cast<memref::StoreOp>(use.getOwner())) {
      // Operand 1 = memref, operand 2 = index (operand 0 = value)
      storeOp.getOperation()->setOperand(1, aggregatedMemrefArg);
      storeOp.getOperation()->setOperand(2, newIndex);
    } else if (auto transferReadOp =
                   dyn_cast<vector::TransferReadOp>(use.getOwner())) {
      // Operand 0 = source memref, operand 1 = index
      transferReadOp.getOperation()->setOperand(0, aggregatedMemrefArg);
      transferReadOp.getOperation()->setOperand(1, newIndex);
    } else {
      assert(false && "unsupported memref use operation");
    }
  }
}

namespace {
class AggregateMemoryPass
    : public dynamatic::impl::AggregateMemoryBase<AggregateMemoryPass> {
  void runDynamaticPass() override {
    mlir::ModuleOp modOp = getOperation();
    int uniqueMemrefId = 0;
    // Identify all funcOp
    auto funcOps = modOp.getOps<func::FuncOp>();
    for (auto funcOp : funcOps) {
      // Collect memref by element type saved
      std::vector<std::pair<Type, std::vector<Value>>> memrefByType =
          groupMemrefsByType(funcOp);

      // For each memref type with more than one memref
      for (auto &memrefTypeAndValues : memrefByType) {
        auto elementType = memrefTypeAndValues.first;
        auto &memrefValues = memrefTypeAndValues.second;
        if (memrefValues.size() <= 1)
          continue; // Nothing to aggregate

        auto aggregatedMemrefArg = createAggregegatedMemrefArg(
            funcOp, elementType, memrefValues, uniqueMemrefId);
        uniqueMemrefId++;

        // Replace all uses of the original memrefs with slices of the new
        // aggregated memref
        int offsetMemref = 0;
        for (auto memrefValue : memrefValues) {
          // For each use of the original memref, find the index and replace
          // it with a new index into the aggregated memref
          auto memRefType = memrefValue.getType().cast<MemRefType>();
          replaceUsesOfMemrefIntoAggregatedMemref(
              funcOp, memrefValue, aggregatedMemrefArg, offsetMemref);
          // Update offset for next memref
          int memrefSize = memRefType.getNumElements();
          offsetMemref += memrefSize;
        }
        // Remove the original memref arguments from the function
        for (auto memrefValue : memrefValues) {
          int argIndex = memrefValue.cast<BlockArgument>().getArgNumber();
          funcOp.eraseArgument(argIndex);
        }
        // Change the function type to reflect the new arguments
        SmallVector<Type> newArgTypes;
        for (auto arg : funcOp.getArguments()) {
          newArgTypes.push_back(arg.getType());
        }
        funcOp.setType(FunctionType::get(funcOp.getContext(), newArgTypes,
                                         funcOp.getResultTypes()));
      }
    }
  }
};
} // namespace

namespace dynamatic {
std::unique_ptr<dynamatic::DynamaticPass> createAggregateMemoryPass() {
  return std::make_unique<AggregateMemoryPass>();
}
} // namespace dynamatic