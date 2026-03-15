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
/// Written during Step 1 (unbundling) and consumed during Step 3 (populate).
static mlir::DenseMap<mlir::Operation *, std::string> opToBlifPathMap;

/// Port name for the clock signal added to every rewritten hw module.
static const std::string clockSignal = "clk";

/// Port name for the reset signal added to every rewritten hw module.
static const std::string resetSignal = "rst";

/// Identifies which component of a bundled Handshake channel a flat HW port
/// corresponds to (used in Step 1).
enum SignalKind { DATA_SIGNAL = 0, VALID_SIGNAL = 1, READY_SIGNAL = 2 };

//===----------------------------------------------------------------------===//
// Step 2 - Sub-component A: PortLayout
//
// Describes the new port list that results from applying the ready-direction
// flip and bit-splitting to an old hw::HWModuleOp. Callers use the result to
// create new modules and to seed the SignalTracker.
//===----------------------------------------------------------------------===//

/// Describes one flat i1 port in the rewritten module.
struct RewrittenPort {
  std::string name; ///< final port name (bit-indexed if needed)
  hw::ModulePort::Direction direction; ///< direction in the *new* module
  unsigned newIndex; ///< position in the new module's port list

  /// The Value in the *old* module that this port corresponds to
  mlir::Value oldSignal;

  /// True only for non-ready ports that were outputs in the old module.
  /// Used to populate the oldOutputIdx -> newOutputIdx index table.
  bool wasNonReadyOutput = false;
  unsigned oldOutputIdx = 0; ///< valid only when wasNonReadyOutput
};

/// The complete result of computePortLayout(): everything needed to create
/// the new hw::HWModuleOp and to seed the SignalTracker.
struct PortLayout {
  llvm::SmallVector<RewrittenPort> ports; ///< one entry per flat i1 port
  unsigned clkNewIndex;                   ///< port index of the added clk input
  unsigned rstNewIndex;                   ///< port index of the added rst input
};

/// Analyses \p oldMod's port list and returns a PortLayout describing the
/// new module's ports.
///
/// Direction rules:
///   ready input   ->  output  (flipped)
///   ready output  ->  input   (flipped)
///   non-ready input  ->  input   (unchanged)
///   non-ready output ->  output  (unchanged)
///
/// Multi-bit ports are expanded into individual i1 ports.
/// clk and rst index slots are recorded in the layout; the actual
/// hw::PortInfo entries are added by buildModulePortInfo().
PortLayout computePortLayout(hw::HWModuleOp oldMod);

/// Converts \p layout into the hw::ModulePortInfo needed to construct the
/// new hw::HWModuleOp.  clk and rst inputs are appended automatically.
hw::ModulePortInfo buildModulePortInfo(const PortLayout &layout,
                                       mlir::MLIRContext *ctx);

//===----------------------------------------------------------------------===//
// Step 2 - Sub-component B: SignalTracker
//
// Maintains the mapping from old-module Values to new-module Values as
// modules are rewritten one by one.
//
// Intended three-phase usage per module:
//   1. recordInputs()          - seed with the new module's block arguments
//   2. resolve()               - look up (or create placeholders for) operand
//                                values when building new hw::InstanceOps
//   3. commit()                - record definitive result values and patch
//                                any outstanding placeholders
//===----------------------------------------------------------------------===//

class SignalTracker {
public:
  // -----------------------------------------------------------------------
  // Seeding - called when a new hw::HWModuleOp has just been created
  // -----------------------------------------------------------------------

  /// Records the mapping from each new input block argument of \p newMod back
  /// to the corresponding old-module Value described by \p layout.
  void recordInputs(const PortLayout &layout, hw::HWModuleOp newMod);

  /// Captures the clk and rst block arguments of the top module.
  /// Must be called exactly once, for the top-level module only.
  void recordClkRst(const PortLayout &layout, hw::HWModuleOp newMod);

  /// Stores the oldOutputIdx -> newOutputIdx mapping for \p modName so that
  /// commit() can locate the right result slots on new instances.
  void recordOutputIndexMap(
      mlir::StringAttr modName,
      llvm::SmallVector<std::pair<unsigned, llvm::SmallVector<unsigned>>>
          mapping);

  // -----------------------------------------------------------------------
  // Resolution - called while building hw::InstanceOp operands
  // -----------------------------------------------------------------------

  /// Returns the new-module Value(s) corresponding to \p oldSignal.
  ///
  /// If no resolved mapping exists yet a hw::ConstantOp(0) placeholder is
  /// inserted for each bit; commit() will replace it with the real Value.
  llvm::SmallVector<mlir::Value>
  resolve(mlir::Value oldSignal, mlir::OpBuilder &builder, mlir::Location loc);

  // -----------------------------------------------------------------------
  // Committing - called after a new hw::InstanceOp has been created
  // -----------------------------------------------------------------------

  /// Records that \p oldResult maps to the corresponding result(s) of
  /// \p newInst, and patches any placeholders previously inserted for it.
  ///
  /// Pass \p oldOutputIdx = -1 for ready signals (no index-based mapping).
  void commit(mlir::Value oldResult, llvm::StringRef portName, int oldOutputIdx,
              hw::HWModuleOp oldMod, hw::HWModuleOp newMod,
              hw::InstanceOp newInst);

  // -----------------------------------------------------------------------
  // Accessors
  // -----------------------------------------------------------------------

  /// Returns the clock Value for the top module.  Asserts it has been set.
  mlir::Value getClk() const;

  /// Returns the reset Value for the top module.  Asserts it has been set.
  mlir::Value getRst() const;

  /// Returns the resolved Value(s) for \p oldSignal.
  /// Asserts a definitive mapping already exists (i.e. after commit()).
  llvm::SmallVector<mlir::Value> get(mlir::Value oldSignal) const;

private:
  /// Finds the result indices on \p newInst that correspond to \p oldOutputIdx.
  /// Falls back to port-name lookup for ready signals (oldOutputIdx == -1).
  llvm::SmallVector<unsigned> findNewResultIndices(llvm::StringRef portName,
                                                   int oldOutputIdx,
                                                   hw::HWModuleOp oldMod,
                                                   hw::HWModuleOp newMod);

  mlir::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>> resolvedMap;
  mlir::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>> placeholderMap;
  mlir::DenseMap<
      mlir::StringAttr,
      llvm::SmallVector<std::pair<unsigned, llvm::SmallVector<unsigned>>>>
      outputIdxMap;

  mlir::Value clk; ///< clk argument of the rewritten top module
  mlir::Value rst; ///< rst argument of the rewritten top module
};

//===----------------------------------------------------------------------===//
// Step 2 - Sub-component C: ModuleRewriter
//
// Orchestrates PortLayout (A) and SignalTracker (B).  Contains three focused
// methods so that rewriteModule() is a readable high-level coordinator:
//
//   rewriteModule()    builds the new hw::HWModuleOp, seeds the tracker,
//                      dispatches the body (instances or leaf), and wires
//                      the terminator.
//
//   rewriteInstance()  classifies ports, resolves operands via the tracker,
//                      creates the new hw::InstanceOp, and commits results.
//
//   wireTerminator()   collects resolved output Values from the tracker and
//                      sets them on the new module's hw::OutputOp.
//===----------------------------------------------------------------------===//

class ModuleRewriter {
public:
  explicit ModuleRewriter(llvm::StringRef topFunctionName);

  /// Rewrites \p oldMod and stores the result in \p newMods.
  /// Recursively rewrites any modules referenced by instances inside
  /// \p oldMod before processing it.
  void rewriteModule(hw::HWModuleOp oldMod, mlir::ModuleOp parent,
                     mlir::SymbolTable &symTable,
                     mlir::DenseMap<llvm::StringRef, hw::HWModuleOp> &newMods,
                     mlir::DenseMap<llvm::StringRef, hw::HWModuleOp> &oldMods);

private:
  /// Rewrites a single hw::InstanceOp from the old body into the new module.
  void
  rewriteInstance(hw::InstanceOp oldInst, mlir::ModuleOp parent,
                  mlir::SymbolTable &symTable,
                  mlir::DenseMap<llvm::StringRef, hw::HWModuleOp> &newMods,
                  mlir::DenseMap<llvm::StringRef, hw::HWModuleOp> &oldMods);

  /// Sets the operands of \p newMod's hw::OutputOp using \p layout and the
  /// tracker's resolved output Values.
  void wireTerminator(const PortLayout &layout, hw::HWModuleOp newMod);

  SignalTracker tracker;
  llvm::StringRef topName;
};

//===----------------------------------------------------------------------===//
// Step 2 - Public entry point: SignalRewriter
//
// The only class the pass itself needs to interact with for Step 2.
// Wraps ModuleRewriter and owns the top-function name.
//===----------------------------------------------------------------------===//

class SignalRewriter {
public:
  /// Rewrites every hw::HWModuleOp in \p modOp to invert ready-signal
  /// directions, split multi-bit ports to i1, and add clk/rst.
  /// Erases the originals and renames the rewritten copies back to the
  /// original names so the rest of the pipeline is unaffected.
  mlir::LogicalResult rewriteAllSignals(mlir::ModuleOp modOp);

  /// Sets the name of the top-level Handshake / HW function.
  /// Must be called before rewriteAllSignals().
  void setTopFunctionName(llvm::StringRef name) {
    assert(!name.empty() && "top function name must not be empty");
    topFunction = name;
  }

  /// Returns the stored top function name.  Asserts it has been set.
  llvm::StringRef getTopFunctionName() const {
    assert(!topFunction.empty() && "top function name has not been set");
    return topFunction;
  }

private:
  llvm::StringRef topFunction;
};

} // namespace dynamatic