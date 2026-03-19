//===- HandshakeToSynth.h - Convert func/cf to handhsake dialect ---*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the helper class for performing the lowering of the
// --lower-handshake-to-synth conversion pass.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/HW/HWOpInterfaces.h"
#include "dynamatic/Dialect/HW/HWOps.h"
#include "dynamatic/Dialect/HW/HWTypes.h"
#include "dynamatic/Dialect/HW/PortImplementation.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Dialect/Synth/SynthDialect.h"
#include "dynamatic/Dialect/Synth/SynthOps.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/BLIFFileManager.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/BlifImporter/BlifImporterSupport.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/RTL/RTL.h"
#include "dynamatic/Support/Utils/Utils.h"
#include "dynamatic/Transforms/FuncMaximizeSSA.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <bitset>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>

namespace dynamatic {

//===----------------------------------------------------------------------===//
// Pass-wide shared state
//===----------------------------------------------------------------------===//

/// Maps each hw::HWModuleOp (by Operation*) to the BLIF file path it should
/// be populated from.  An empty string means the module needs no replacement.
/// Written during Step 1 (unbundling) and consumed during Step 2 (populate).
static mlir::DenseMap<mlir::Operation *, std::string> opToBlifPathMap;

//===----------------------------------------------------------------------===//
// Step 1: Unbundle handshake channels into individual bits and create
// placeholder hw modules
//===----------------------------------------------------------------------===//

// Add new type for the tuple
using UnbundledValuesTuple = std::tuple<SmallVector<Value>, Value, Value>;

// Represents a single unbundled bit of a data signal
struct DataPortInfo {
  unsigned bitIndex;  // which bit of the original channel
  unsigned totalBits; // total bits in the channel
};

// Represents a valid signal (no extra fields needed)
struct ValidPortInfo {};

// Represents a ready signal (no extra fields needed)
struct ReadyPortInfo {};

using PortKind = std::variant<DataPortInfo, ValidPortInfo, ReadyPortInfo>;

// Struct to hold the new unbundled port information
struct HandshakeUnitPort {
  std::string name;
  // Direction in the *new* module after unbundling (ready signals are flipped).
  hw::ModulePort::Direction direction;
  // The Value in the *old* handshake module that this port corresponds to
  Value handshakeSignal;
  PortKind kind; // additional info about the port, e.g. which bit of the
                 // original channel it corresponds to if it's a data port

  HandshakeUnitPort(std::string name, hw::ModulePort::Direction direction,
                    Value handshakeSignal, PortKind kind)
      : name(std::move(name)), direction(direction),
        handshakeSignal(handshakeSignal), kind(kind) {}
};

// Class that controls the unbundling of handshake channel types into integer
// types
class HandshakeUnbundler {
public:
  HandshakeUnbundler(ModuleOp modOp)
      : modOp(modOp), symTable(modOp), builder(modOp) {}

  // Top-level entry point for unbundling funcOp and all ops inside it
  mlir::LogicalResult unbundleHandshakeChannels();

private:
  // Converts one Handshake op inside the funcop into an hw instance inside the
  // top function
  mlir::LogicalResult convertHandshakeOp(Operation *op);

  // Converts one Handshake function into an hw module
  mlir::LogicalResult convertHandshakeFunc();

  // Function that returns unbundled values from a channel value. If the channel
  // value has not been unbundled yet, creates backedge placeholders for each
  // bit and saves them.
  llvm::SmallVector<Value> getUnbundledValues(Value handshakeSignal,
                                              unsigned totalBits,
                                              PortKind portKind, Location loc);

  // Function that saves the mapping from a channel value to its unbundled bit
  // values. If there were placeholders for the channel value, replaces them
  // with the real bit values.
  void saveUnbundledValues(Value handshakeSignal, PortKind portKind,
                           llvm::SmallVector<Value> unbundledValues);

  // Helper function to update the old tuple of unbundled values
  UnbundledValuesTuple updateTuple(UnbundledValuesTuple oldTuple,
                                   SmallVector<Value> newValues,
                                   PortKind portKind);

  // Helper function to extract the relevant values from a tuple based on the
  // bit type
  SmallVector<Value> getValuesFromTuple(UnbundledValuesTuple valTuple,
                                        PortKind portKind);

  // Module op
  ModuleOp modOp;
  handshake::FuncOp topFunction;
  // Symbol table for looking up functions and modules
  SymbolTable symTable;
  // Builder for creating new operations
  OpBuilder builder;
  // Top HW Module
  hw::HWModuleOp topHWModule;

  // Maps handshake channel values to their unbundled bit values. The tuple is
  // data bits, valid bit, ready bit
  llvm::DenseMap<Value, UnbundledValuesTuple> unbundledValuesMap;
  // Maps handshake channel values to any placeholder backedges created for
  // their unbundled bits. The tuple is data bits, valid bit, ready bit
  llvm::DenseMap<Value, UnbundledValuesTuple> pendingValuesMap;

  // clk and rst values from the top module
  Value clk;
  Value rst;

  // Backedge builder for creating placeholders for unbundled bits
  std::unique_ptr<BackedgeBuilder> backedgeBuilder;
};

//===----------------------------------------------------------------------===//
// Step 2: Populate the hw modules with BLIF content
//===----------------------------------------------------------------------===//

// Top function for this step. For each hw module created in Step 1, look up the
// corresponding BLIF file and populate the module body with the content of the
// BLIF file.
mlir::LogicalResult populateAllHWModules(mlir::ModuleOp modOp,
                                         llvm::StringRef topModuleName);

// Function that populates one hw module by replacing its body with the content
// of the corresponding BLIF file
mlir::LogicalResult
populateHWModule(mlir::ModuleOp modOp, hw::InstanceOp inst,
                 llvm::DenseSet<StringRef> &populatedModules);

} // namespace dynamatic