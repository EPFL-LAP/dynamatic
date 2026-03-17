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
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/BlifImporter/BlifImporterSupport.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
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
#include <regex>
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

/// Port name for the clock signal added to every rewritten hw module.
static const std::string clockSignal = "clk";

/// Port name for the reset signal added to every rewritten hw module.
static const std::string resetSignal = "rst";

//===----------------------------------------------------------------------===//
// Step 2: BlifPopulator
//
// Replaces the synth::SubcktOp placeholder body of each hw::HWModuleOp with
// the gate-level netlist imported from its BLIF file.
//
// Owns the SymbolTable (built once at construction) and the deduplication
// set (so populate() can be called once per hw::InstanceOp without worrying
// about processing the same module definition twice).
//===----------------------------------------------------------------------===//

class BlifPopulator {
public:
  /// Constructs a populator for \p modOp, building the SymbolTable once.
  explicit BlifPopulator(mlir::ModuleOp modOp);

  /// Replaces the body of the hw::HWModuleOp referenced by \p inst with the
  /// BLIF netlist stored in opToBlifPathMap.
  ///
  /// Returns success() immediately if:
  ///   - the module has already been populated, or
  ///   - the recorded BLIF path is empty (module needs no replacement).
  /// Returns failure() if the module cannot be found or the import fails.
  mlir::LogicalResult populate(hw::InstanceOp inst);

private:
  mlir::ModuleOp modOp;
  mlir::SymbolTable symTable;

  /// Names of hw modules that have already been populated, used to skip
  /// duplicate processing when multiple instances reference the same module.
  llvm::DenseSet<mlir::StringAttr> done;
};

/// Walks every hw::InstanceOp inside the top-level hw module named
/// \p topModuleName and calls BlifPopulator::populate() for each one.
mlir::LogicalResult populateAllHWModules(mlir::ModuleOp modOp,
                                         llvm::StringRef topModuleName);

// Add new type for the tuple
using UnbundledValuesTuple = std::tuple<SmallVector<Value>, Value, Value>;

// Port bit types
enum PortBitType { DATA, VALID, READY };

// Struct to hold the new unbundled port information
struct UnbundledPort {
  std::string name;
  // Direction in the *new* module after unbundling (ready signals are flipped).
  hw::ModulePort::Direction direction;
  // The Value in the *old* handshake module that this port corresponds to
  Value handshakeSignal;
  unsigned bitIndex;  // for data signals, indicates which bit of the original
                      // channel this port corresponds to
  unsigned totalBits; // for data signals, indicates the total number of bits in
                      // the original channel (used for naming)
  PortBitType bitType; // indicates whether this port corresponds to data,
                       // valid, or ready component of the original channel
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
  // value has not been unbundled yet, creates hw constant placeholders for each
  // bit and saves them.
  llvm::SmallVector<Value> getUnbundledValues(Value channelVal,
                                              unsigned totalBits,
                                              PortBitType bitType,
                                              Location loc);

  // Function that saves the mapping from a channel value to its unbundled bit
  // values. If there were placeholders for the channel value, replaces them
  // with the real bit values.
  void saveUnbundledValues(Value channelVal, PortBitType bitType,
                           llvm::SmallVector<Value> bitValues);

  // Helper function to format tuple from map
  UnbundledValuesTuple updateTuple(UnbundledValuesTuple oldTuple,
                                   SmallVector<Value> newValues,
                                   PortBitType bitType);

  SmallVector<Value> getValuesFromTuple(UnbundledValuesTuple valTuple,
                                        PortBitType bitType);

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
  // Maps handshake channel values to any placeholder constants created for
  // their unbundled bits. The tuple is data bits, valid bit, ready bit
  llvm::DenseMap<Value, UnbundledValuesTuple> placeholderMap;

  // clk and rst values from the top module
  Value clk;
  Value rst;
};

} // namespace dynamatic