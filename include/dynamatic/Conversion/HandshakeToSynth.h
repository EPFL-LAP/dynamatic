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

// Keywords for data and control signals when unbundling Handshake types using
// enums.
enum SignalKind {
  DATA_SIGNAL = 0,
  VALID_SIGNAL = 1,
  READY_SIGNAL = 2,
};

// Strings representing the name of the clock and reset signals
static const std::string clockSignal = "clk";
static const std::string resetSignal = "rst";

// Utility class to invert the ready signals and add reset and clock signals to
// the HW modules created from Handshake operations
class SignalRewriter {
public:
  /// Rewrite a HW module interface and body to apply signal-direction and
  /// bit-level rewrites (e.g., ready inversion and bit unbundling).
  void rewriteHWModule(hw::HWModuleOp oldMod, ModuleOp parent,
                       SymbolTable &symTable,
                       DenseMap<StringRef, hw::HWModuleOp> &newHWModules,
                       DenseMap<StringRef, hw::HWModuleOp> &oldHWModules);

  /// Rewrite a HW instance to use a rewritten module and updated operands.
  void rewriteHWInstance(hw::InstanceOp oldInst, ModuleOp parent,
                         SymbolTable &symTable,
                         DenseMap<StringRef, hw::HWModuleOp> &newHWModules,
                         DenseMap<StringRef, hw::HWModuleOp> &oldHWModules);

  /// Get the name of the rewritten HW module (e.g., by appending a suffix).
  mlir::StringAttr getRewrittenModuleName(hw::HWModuleOp oldMod,
                                          mlir::MLIRContext *ctx);

  /// Get the mapping of an old input signal to its rewritten signals.
  SmallVector<Value> getInputSignalMapping(Value oldInputSignal,
                                           OpBuilder builder, Location loc);

  /// Update the mapping for an output signal group after a new instance is
  /// created.
  void updateOutputSignalMapping(Value oldResult, StringRef outputName,
                                 int oldOutputIdx, hw::HWModuleOp oldMod,
                                 hw::HWModuleOp newMod, hw::InstanceOp newInst);

  /// Rewrite all HW modules in the given MLIR module to apply signal
  /// restructuring (direction changes, bit unbundling, etc.).
  LogicalResult rewriteAllSignals(mlir::ModuleOp modOp);

  // Function to set the name of the top function
  void setTopFunctionName(StringRef topFunctionName) {
    assert(topFunctionName != "" &&
           "top function name cannot be set to an empty string");
    topFunction = topFunctionName;
  }

  // Function to get the name of the top function
  StringRef getTopFunctionName() {
    assert(
        topFunction != "" &&
        "top function name should be set before being able to get its value");
    return topFunction;
  }

private:
  // IMPORTANT: A fundamental assumption for these maps to work is that each
  // handshake channel connects uniquely one handshake unit to another one. If
  // this assumption is broken, the maps will not work correctly since one key
  // could correspond to multiple values.
  //
  // Maps to keep track of signal connections during instance rewriting.
  //
  // "Old" refers to the original HW instance, with potentially incorrect
  // signal directions and HW types that may have bitwidths larger than 1.
  // "New" refers to the rewritten HW instance, with corrected signal
  // directions and HW types normalized to 1-bit signals for each component.
  DenseMap<Value, SmallVector<Value>> oldModuleSignalToNewModuleSignalsMap;

  // Since the HW instances are rewritten in a recursive manner, it might be
  // possible that we need results of an instance that has not been created
  // yet. To handle this case, we create temporary HW constant operations
  // to hold places for these values. This map keeps track of these temporary
  // values grouped per original signal.
  DenseMap<Value, SmallVector<Value>> oldModuleSignalToTempValuesMap;

  // Map to connect the idx of the old output signal to the idxs of the new
  // output signals after rewriting.
  DenseMap<StringAttr, SmallVector<std::pair<unsigned, SmallVector<unsigned>>>>
      oldOutputIdxToNewOutputIdxMap;

  // Name of the top function
  StringRef topFunction = "";
  // Signal of the top function that refers to the clock signal of the top func
  Value clkSignalTop;
  // Signal of the top function that refers to the reset signal of the top func
  Value rstSignalTop;
};

} // namespace dynamatic