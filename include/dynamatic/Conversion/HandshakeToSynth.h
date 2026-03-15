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

// ===----------------------------------------------------------------------===//
// Signal kind enum
// ===----------------------------------------------------------------------===//

/// Identifies which component of a bundled Handshake channel a flat HW port
/// corresponds to.
enum SignalKind {
  DATA_SIGNAL = 0,
  VALID_SIGNAL = 1,
  READY_SIGNAL = 2,
};

// ===----------------------------------------------------------------------===//
// Global state shared across pass steps
// ===----------------------------------------------------------------------===//

/// Maps each hw::HWModuleOp (by Operation*) to the BLIF file path it should
/// be populated from. An empty string means the module needs no replacement.
/// Populated during Step 1 (unbundling) and consumed during Step 3.
static mlir::DenseMap<mlir::Operation *, std::string> opToBlifPathMap;

/// Port name string for the clock signal added to all rewritten hw modules.
static const std::string clockSignal = "clk";

/// Port name string for the reset signal added to all rewritten hw modules.
static const std::string resetSignal = "rst";

// ===----------------------------------------------------------------------===//
// SignalRewriter: Step 2 - invert ready signal directions and unbundle bits
// ===----------------------------------------------------------------------===//

/// Rewrites all HW modules produced in Step 1 to comply with the standard
/// handshake convention:
///   - ready signals travel in the opposite direction to data/valid signals
///   - all multi-bit signals are split into individual i1 ports
///   - clock and reset ports are added to every module
///
/// The rewriter works recursively: rewriting a module first rewrites any
/// hw::HWModuleOps it instantiates, so that new instances can reference
/// already-corrected module interfaces.
class SignalRewriter {

public:
  /// Entry point: rewrites every hw::HWModuleOp in \p modOp, then removes
  /// the originals and renames the rewritten copies back to the original
  /// names so that the rest of the pipeline is unaffected.
  mlir::LogicalResult rewriteAllSignals(mlir::ModuleOp modOp);

  /// Sets the name of the top-level Handshake function. Must be called before
  /// rewriteAllSignals(); the top module is special-cased to capture clk/rst.
  void setTopFunctionName(llvm::StringRef name) {
    assert(!name.empty() && "top function name must not be empty");
    topFunction = name;
  }

  /// Returns the top function name (asserts it has been set).
  llvm::StringRef getTopFunctionName() const {
    assert(!topFunction.empty() && "top function name has not been set");
    return topFunction;
  }

private:
  // -----------------------------------------------------------------------
  // Core rewriting helpers
  // -----------------------------------------------------------------------

  /// Rewrite a HW module interface and body to apply signal-direction and
  /// bit-level rewrites (e.g., ready inversion and bit unbundling).
  void rewriteHWModule(hw::HWModuleOp oldMod, ModuleOp parent,
                       SymbolTable &symTable,
                       DenseMap<StringRef, hw::HWModuleOp> &newHWModules,
                       DenseMap<StringRef, hw::HWModuleOp> &oldHWModules);

  /// Rewrites a single hw::InstanceOp inside a module that is being rewritten.
  /// Looks up (or triggers) the rewrite of the referenced module, then builds
  /// a new instance with corrected operand/result connections.
  void rewriteHWInstance(hw::InstanceOp oldInst, ModuleOp parent,
                         SymbolTable &symTable,
                         DenseMap<StringRef, hw::HWModuleOp> &newHWModules,
                         DenseMap<StringRef, hw::HWModuleOp> &oldHWModules);

  // -----------------------------------------------------------------------
  // Signal mapping helpers
  // -----------------------------------------------------------------------

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

  // -----------------------------------------------------------------------
  // Internal state
  // -----------------------------------------------------------------------

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

  /// Name of the top-level Handshake function/hw module.
  llvm::StringRef topFunction;

  /// Block argument of the new top module that carries the clock signal.
  mlir::Value clkSignalTop;

  /// Block argument of the new top module that carries the reset signal.
  mlir::Value rstSignalTop;
};

} // namespace dynamatic