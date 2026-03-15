//===- HandshakeToSynth.cpp - Convert Handshake to Synth --------------*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --handshake-to-synth conversion pass in three steps:
//
//   Step 1 - Unbundle: convert every Handshake op into an hw::HWModuleOp
//            whose ports are flat integer signals (data/valid/ready split),
//            connected through unrealized-conversion casts that are later
//            eliminated.
//
//   Step 2 - Rewrite signals: invert the direction of all ready signals
//            (ready travels opposite to data/valid), split multi-bit ports
//            into individual i1 ports, and add clk/rst to every module.
//
//   Step 3 - Populate: replace each hw::HWModuleOp's placeholder body with
//            the actual gate-level netlist imported from the BLIF file whose
//            path was recorded during Step 1.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Conversion/HandshakeToSynth.h"

// [START Boilerplate code for the MLIR pass]
#include "dynamatic/Conversion/Passes.h" // IWYU pragma: keep
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
namespace dynamatic {
#define GEN_PASS_DEF_HANDSHAKETOSYNTH
#include "dynamatic/Conversion/Passes.h.inc"
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

//===----------------------------------------------------------------------===//
// Step 1 — Unbundle Handshake types into flat HW ports
//===----------------------------------------------------------------------===//
//
// Converts every Handshake operation inside a handshake::FuncOp into an
// hw::HWModuleOp whose ports are flat i1/iN signals (data / valid / ready
// split apart).  The conversion is wired together with temporary
// UnrealizedConversionCastOps that are eliminated at the end of this step.
//
// The code is organised in four layers, each depending only on the layers
// above it:
//
//   Layer 1 — TypeUnbundler   : knows how to split a Handshake type into its
//                               component (SignalKind, Type) pairs.
//
//   Layer 2 — PortInfoBuilder : knows how to derive port names from a
//                               Handshake op and zip them with unbundled types
//                               to produce hw::ModulePortInfo.
//
//   Layer 3 — ConversionPatterns : the TypeConverter and OpConversionPatterns
//                               that drive the DialectConversion framework.
//                               Also contains the cast helpers and the synth
//                               placeholder installer.
//
//   Layer 4 — Orchestrator   : unbundleAllHandshakeTypes(), the single entry
//                               point of this step.  Runs the three
//                               sequential phases and nothing else.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
// Layer 1 — TypeUnbundler
//
// Pure functions with no side effects and no dependencies on other code in
// this file. They determine how a Handshake type expands into flat hardware
// types and specify the role associated with each resulting signal.
//
//===----------------------------------------------------------------------===//

/// Splits a single Handshake type into its constituent (SignalKind, Type)
/// pairs:
///   ChannelType  -> { DATA, dataType }, { VALID, i1 }, { READY, i1 }
///   ControlType  ->                     { VALID, i1 }, { READY, i1 }
///   MemRefType   -> { DATA, elemType }, { VALID, i1 }, { READY, i1 }
///   anything else-> { DATA, type }   (pass-through)
SmallVector<std::pair<SignalKind, Type>> unbundleType(Type type) {
  return TypeSwitch<Type, SmallVector<std::pair<SignalKind, Type>>>(type)
      // Case for channel type where we unbundle into data, valid, ready
      .Case<handshake::ChannelType>([](handshake::ChannelType chanType) {
        return SmallVector<std::pair<SignalKind, Type>>{
            std::make_pair(DATA_SIGNAL, chanType.getDataType()),
            std::make_pair(VALID_SIGNAL,
                           IntegerType::get(chanType.getContext(), 1)),
            std::make_pair(READY_SIGNAL,
                           IntegerType::get(chanType.getContext(), 1))};
      })
      // Case for control type where we unbundle into valid, ready
      .Case<handshake::ControlType>([](handshake::ControlType ctrlType) {
        return SmallVector<std::pair<SignalKind, Type>>{
            std::make_pair(VALID_SIGNAL,
                           IntegerType::get(ctrlType.getContext(), 1)),
            std::make_pair(READY_SIGNAL,
                           IntegerType::get(ctrlType.getContext(), 1))};
      })
      // Case for memref type where we extract the element type as data,
      // valid, ready
      .Case<MemRefType>([](MemRefType memType) {
        return SmallVector<std::pair<SignalKind, Type>>{
            std::make_pair(DATA_SIGNAL, memType.getElementType()),
            std::make_pair(VALID_SIGNAL,
                           IntegerType::get(memType.getContext(), 1)),
            std::make_pair(READY_SIGNAL,
                           IntegerType::get(memType.getContext(), 1))};
      })
      .Default([&](Type t) {
        return SmallVector<std::pair<SignalKind, Type>>{
            std::make_pair(DATA_SIGNAL, t)};
      });
}

/// Function that returns only the flat Types (dropping the SignalKind tag).
static SmallVector<Type> unbundleTypeFlat(Type type) {
  SmallVector<Type> result;
  for (auto [_, t] : unbundleType(type))
    result.push_back(t);
  return result;
}

/// Function that fills \p inputPorts and \p outputPorts with the unbundled
/// (SignalKind,Type) groups for every operand/result of \p op.
/// handshake::FuncOp is handled specially because its "operands" are block
/// arguments rather than SSA uses.
void unbundleOpPorts(
    Operation *op,
    SmallVector<SmallVector<std::pair<SignalKind, Type>>> &inputPorts,
    SmallVector<SmallVector<std::pair<SignalKind, Type>>> &outputPorts) {
  // If the operation is a handshake function, the ports are extracted
  // differently
  if (isa<handshake::FuncOp>(op)) {
    handshake::FuncOp funcOp = cast<handshake::FuncOp>(op);
    // Extract ports from the function type
    for (auto arg : funcOp.getArguments()) {
      inputPorts.push_back(unbundleType(arg.getType()));
    }
    for (auto resultType : funcOp.getResultTypes()) {
      outputPorts.push_back(unbundleType(resultType));
    }
    return;
  }
  for (auto input : op->getOperands()) {
    // Unbundle the input type depending on its actual type
    inputPorts.push_back(unbundleType(input.getType()));
  }
  for (auto result : op->getResults()) {
    // Unbundle the output type depending on its actual type
    outputPorts.push_back(unbundleType(result.getType()));
  }
}

//===----------------------------------------------------------------------===//
// Layer 2 — PortInfoBuilder
//
// Knows how to turn a Handshake op into an hw::ModulePortInfo.
// Uses Layer 1 for type splitting; uses the NamedIOInterface for names.
// Two private helpers handle the string formatting; the single public
// entry point is buildPortInfo().
//===----------------------------------------------------------------------===//

// ---------------------------------------------------------------------------
// Private string helpers
// ---------------------------------------------------------------------------

/// Inserts \p suffix immediately before the first '[' in \p name, or appends
/// it at the end if no '[' is found.
/// Example: insertSuffix("data[3]", "_valid") → "data_valid[3]"
std::string insertSuffix(const std::string &name, const std::string &suffix) {
  std::size_t pos = name.find('[');
  if (pos != std::string::npos) {
    std::string result = name;
    result.insert(pos, suffix);
    return result;
  }
  return name + suffix;
}

/// Converts a (root, index) pair into the canonical "root[index]" form.
/// If \p root already contains "[N]", the old index is linearised with
/// \p arrayWidth before adding \p index.
/// Example: formatArrayName("data[2]", 3, 4) becomes "data[11]"  (2*4 + 3)
std::string formatArrayName(const std::string &root, unsigned index,
                            unsigned arrayWidth = 0) {
  static const std::regex arrayPattern(R"((\w+)\[(\d+)\])");
  std::smatch m;
  if (std::regex_match(root, m, arrayPattern)) {
    assert(arrayWidth != 0 && "arrayWidth required for already-indexed names");
    unsigned linearised = std::stoi(m[2].str()) * arrayWidth + index;
    return m[1].str() + "[" + std::to_string(linearised) + "]";
  }
  return root + "[" + std::to_string(index) + "]";
}

// ---------------------------------------------------------------------------
// Private: derive per-operand/result base names from a Handshake op
// ---------------------------------------------------------------------------

/// Returns the base port names for every operand and result of \p op,
/// in the "root[idx]" format expected by BLIF.
///
/// The Handshake NamedIOInterface produces names like "in_0", "in_1".
/// This function converts them:
///   - "in_0" alone stays as "in"    (no sibling with idx > 0 yet)
///   - once "in_1" appears, "in" is back-patched to "in[0]",
///     and "in_1" becomes "in[1]"
///
/// handshake::FuncOp has no NamedIOInterface so it gets generic names.
std::pair<SmallVector<std::string>, SmallVector<std::string>>
buildPortNames(Operation *op) {
  SmallVector<std::string> inputNames, outputNames;

  // FuncOp: generic fallback names.
  if (auto funcOp = dyn_cast<handshake::FuncOp>(op)) {
    for (auto [idx, _] : llvm::enumerate(funcOp.getArguments()))
      inputNames.push_back("in" + std::to_string(idx));
    for (auto [idx, _] : llvm::enumerate(funcOp.getResultTypes()))
      outputNames.push_back("out" + std::to_string(idx));
    return {inputNames, outputNames};
  }

  auto namedIO = dyn_cast<handshake::NamedIOInterface>(op);
  if (!namedIO) {
    llvm::errs() << op->getName() << " does not implement NamedIOInterface\n";
    assert(false && "cannot build port names without NamedIOInterface");
  }

  // Pattern "root_N": convert to "root[N]" with back-patching at N == 1.
  static const std::regex indexPattern(R"((\w+)_(\d+))");

  auto elaborateName = [&](StringRef raw, SmallVector<std::string> &names) {
    std::string rawStr = raw.str();
    std::smatch m;
    if (!std::regex_match(rawStr, m, indexPattern)) {
      // No index suffix: keep the name as-is.
      names.push_back(rawStr);
      return;
    }
    std::string root = m[1].str();
    unsigned idx = std::stoi(m[2].str());

    if (idx == 0) {
      // First element: use the bare root for now; may be back-patched later.
      names.push_back(root);
    } else {
      if (idx == 1) {
        // Back-patch the first element from "root" to "root[0]".
        auto it = llvm::find(names, root);
        assert(it != names.end() &&
               "could not find index-0 port to back-patch");
        *it = formatArrayName(root, 0);
      }
      names.push_back(formatArrayName(root, idx));
    }
  };

  for (auto [idx, _] : llvm::enumerate(op->getOperands()))
    elaborateName(namedIO.getOperandName(idx), inputNames);
  for (auto [idx, _] : llvm::enumerate(op->getResults()))
    elaborateName(namedIO.getResultName(idx), outputNames);

  return {inputNames, outputNames};
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Builds and returns the hw::ModulePortInfo for the flat HW module that
/// will represent \p op.
///
/// For each operand/result group the function:
///   1. Looks up the base name via buildPortNames().
///   2. Looks up the unbundled (SignalKind, Type) pairs via unbundleOpPorts().
///   3. Zips them: DATA keeps the base name; VALID appends "_valid"; READY
///      appends "_ready" (both inserted before any "[" to preserve indexing).
hw::ModulePortInfo buildPortInfo(Operation *op) {
  MLIRContext *ctx = op->getContext();
  auto [inputNames, outputNames] = buildPortNames(op);

  SmallVector<SmallVector<std::pair<SignalKind, Type>>> unbundledIn,
      unbundledOut;
  unbundleOpPorts(op, unbundledIn, unbundledOut);

  auto applyKind = [](const std::string &base, SignalKind kind) -> std::string {
    switch (kind) {
    case DATA_SIGNAL:
      return base;
    case VALID_SIGNAL:
      return insertSuffix(base, "_valid");
    case READY_SIGNAL:
      return insertSuffix(base, "_ready");
    }
    llvm_unreachable("unknown SignalKind");
  };

  SmallVector<hw::PortInfo> hwInputs, hwOutputs;
  for (auto [idx, portGroup] : llvm::enumerate(unbundledIn)) {
    for (auto [kind, type] : portGroup) {
      hwInputs.push_back(
          {hw::ModulePort{
               StringAttr::get(ctx, applyKind(inputNames[idx], kind)), type,
               hw::ModulePort::Direction::Input},
           idx});
    }
  }
  for (auto [idx, portGroup] : llvm::enumerate(unbundledOut)) {
    for (auto [kind, type] : portGroup) {
      hwOutputs.push_back(
          {hw::ModulePort{
               StringAttr::get(ctx, applyKind(outputNames[idx], kind)), type,
               hw::ModulePort::Direction::Output},
           idx});
    }
  }
  return hw::ModulePortInfo(hwInputs, hwOutputs);
}

//===----------------------------------------------------------------------===//
// Layer 3 — ConversionPatterns
//
// Contains the TypeConverter and OpConversionPatterns that drive the
// DialectConversion framework.  Also contains the cast helpers and the synth
// placeholder installer.
//
//===----------------------------------------------------------------------===//

// ---------------------------------------------------------------------------
// TypeConverter
// ---------------------------------------------------------------------------

/// Teaches the DialectConversion framework how to convert bundled Handshake
/// types to their flat HW equivalents.  Materialization is pass-through
/// because the cast ops inserted by the conversion patterns handle the actual
/// value bridging.
class ChannelUnbundlingTypeConverter : public TypeConverter {
public:
  ChannelUnbundlingTypeConverter() {
    addConversion([](Type type, SmallVectorImpl<Type> &results) {
      results.append(unbundleTypeFlat(type));
      return success();
    });
    auto passThrough = [](OpBuilder &, Type, ValueRange inputs,
                          Location) -> std::optional<Value> {
      return inputs.size() == 1 ? std::optional<Value>(inputs[0])
                                : std::nullopt;
    };
    addTargetMaterialization(passThrough);
    addSourceMaterialization(passThrough);
  }
};

// ---------------------------------------------------------------------------
// Cast helpers
// ---------------------------------------------------------------------------

/// Splits \p bundledValue into its flat components by inserting an
/// UnrealizedConversionCastOp.  Produces one result per flat type.
UnrealizedConversionCastOp splitBundledValue(Value bundledValue, Location loc,
                                             PatternRewriter &rewriter) {
  return rewriter.create<UnrealizedConversionCastOp>(
      loc, TypeRange(unbundleTypeFlat(bundledValue.getType())), bundledValue);
}

/// For each type in \p originalTypes, groups the corresponding slice of
/// \p flatValues and re-bundles it with an UnrealizedConversionCastOp.
/// Returns one cast per original type.
SmallVector<UnrealizedConversionCastOp>
rebundleFlatValues(TypeRange originalTypes, ArrayRef<Value> flatValues,
                   Location loc, PatternRewriter &rewriter) {
  assert(flatValues.size() >= originalTypes.size());
  SmallVector<UnrealizedConversionCastOp> casts;
  unsigned offset = 0;
  for (Type origType : originalTypes) {
    unsigned numFlat = unbundleTypeFlat(origType).size();
    SmallVector<Value> slice(flatValues.begin() + offset,
                             flatValues.begin() + offset + numFlat);
    casts.push_back(rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{origType}, slice));
    offset += numFlat;
  }
  return casts;
}

// ---------------------------------------------------------------------------
// Synth placeholder installer
// ---------------------------------------------------------------------------

/// Fills the body of a freshly created \p hwModule with a single
/// synth::SubcktOp that consumes all inputs and produces all outputs.
/// Acts as a placeholder until Step 3 replaces it with the real netlist.
LogicalResult instantiateSynthPlaceholder(hw::HWModuleOp hwModule,
                                          ConversionPatternRewriter &rewriter) {
  rewriter.setInsertionPointToStart(hwModule.getBodyBlock());
  SmallVector<Value> inputs(hwModule.getBodyBlock()->getArguments());
  SmallVector<Type> outputTypes;
  for (auto &port : hwModule.getPortList())
    if (port.isOutput())
      outputTypes.push_back(port.type);

  Operation *terminator = hwModule.getBodyBlock()->getTerminator();
  auto subckt = rewriter.create<synth::SubcktOp>(
      terminator->getLoc(), TypeRange(outputTypes), inputs, "synth_subckt");
  terminator->setOperands(subckt.getResults());
  return success();
}

// ---------------------------------------------------------------------------
// FuncOp to hw::HWModuleOp conversion
// ---------------------------------------------------------------------------

/// Converts the top-level \p funcOp into an hw::HWModuleOp by:
///   1. Creating the module with ports built by buildPortInfo().
///   2. Re-bundling the flat block arguments and inlining the function body.
///   3. Splitting the bundled operands of handshake::EndOp and replacing it
///      with hw::OutputOp.
hw::HWModuleOp convertFuncOpToHWModule(handshake::FuncOp funcOp,
                                       ConversionPatternRewriter &rewriter) {
  MLIRContext *ctx = funcOp.getContext();
  auto modOp = funcOp->getParentOfType<mlir::ModuleOp>();
  SymbolTable symbolTable(modOp);
  StringRef uniqueOpName = getUniqueName(funcOp);
  StringAttr moduleName = StringAttr::get(ctx, uniqueOpName);

  // Re-use an existing module if the pattern fires more than once.
  if (auto existing = symbolTable.lookup<hw::HWModuleOp>(moduleName))
    return existing;

  // --- 1. Create the hw module ---
  rewriter.setInsertionPointToStart(modOp.getBody());
  auto hwModule = rewriter.create<hw::HWModuleOp>(funcOp.getLoc(), moduleName,
                                                  buildPortInfo(funcOp));

  // --- 2. Re-bundle flat block args to bundled args expected by the body ---
  Block *modBlock = hwModule.getBodyBlock();
  Operation *terminator = modBlock->getTerminator();
  rewriter.setInsertionPointToStart(modBlock);

  SmallVector<Value> bundledArgs;
  for (auto &cast :
       rebundleFlatValues(funcOp.getArgumentTypes(),
                          SmallVector<Value>(modBlock->getArguments()),
                          funcOp.getLoc(), rewriter)) {
    assert(cast.getNumResults() == 1);
    bundledArgs.push_back(cast.getResult(0));
  }

  // Inline the function body before the hw terminator.
  rewriter.inlineBlockBefore(funcOp.getBodyBlock(), terminator, bundledArgs);

  // --- 3. Replace handshake::EndOp with hw::OutputOp ---
  handshake::EndOp endOp;
  for (Operation &op : *hwModule.getBodyBlock()) {
    if ((endOp = dyn_cast<handshake::EndOp>(op))) {
      break;
    }
  }
  assert(endOp && "Expected handshake.end after inlining");

  SmallVector<Value> flatOutputs;
  for (Value operand : endOp.getOperands()) {
    auto split = splitBundledValue(operand, endOp->getLoc(), rewriter);
    flatOutputs.append(split.getResults().begin(), split.getResults().end());
  }
  rewriter.setInsertionPointToEnd(endOp->getBlock());
  rewriter.replaceOpWithNewOp<hw::OutputOp>(endOp, flatOutputs);
  // Erase the empty hw::OutputOp placeholder that hw::HWModuleOp auto-inserts.
  for (Operation &op : *hwModule.getBodyBlock()) {
    if (auto out = dyn_cast<hw::OutputOp>(op);
        out && out.getOperands().empty()) {
      rewriter.eraseOp(out);
      break;
    }
  }
  // Remove the original handshake function operation
  rewriter.eraseOp(funcOp);

  return hwModule;
}

// ---------------------------------------------------------------------------
// Generic Handshake op to hw::HWModuleOp + hw::InstanceOp conversion
// ---------------------------------------------------------------------------

/// Converts a single inner Handshake \p op into:
///   1. An hw::HWModuleOp definition (created once per unique op name).
///      Its body contains a synth::SubcktOp placeholder.
///      The op's BLIF path is recorded in opToBlifPathMap for Step 3.
///   2. An hw::InstanceOp that instantiates the module in place of \p op.
///      Bundled operands are split with splitBundledValue(); flat instance
///      results are re-bundled with rebundleFlatValues() so downstream ops
///      still see the expected bundled types.

hw::HWModuleOp convertOpToHWModule(Operation *op,
                                   ConversionPatternRewriter &rewriter) {
  MLIRContext *ctx = op->getContext();
  auto modOp = op->getParentOfType<mlir::ModuleOp>();
  SymbolTable symbolTable(modOp);
  StringRef uniqueOpName = getUniqueName(op);
  StringAttr moduleName = StringAttr::get(ctx, uniqueOpName);

  // --- 1. Look up or create the hw module definition ---
  hw::HWModuleOp hwModule = symbolTable.lookup<hw::HWModuleOp>(moduleName);
  if (!hwModule) {
    // Create it
    rewriter.setInsertionPointToStart(modOp.getBody());

    hwModule = rewriter.create<hw::HWModuleOp>(op->getLoc(), moduleName,
                                               buildPortInfo(op));
    // Instantiate synth placeholder in the HW module body
    if (failed(instantiateSynthPlaceholder(hwModule, rewriter)))
      return nullptr;
    // Record the BLIF path for Step 3
    BLIFImplInterface blifIface = dyn_cast<BLIFImplInterface>(op);
    StringAttr blifAttr = blifIface ? blifIface.getBLIFImpl() : StringAttr{};
    opToBlifPathMap[hwModule.getOperation()] = blifAttr.getValue().str();
  }

  // --- 2. Build flat operands for the instance ---
  rewriter.setInsertionPointAfter(op);
  SmallVector<Value> flatOperands;
  for (auto operand : op->getOperands()) {
    // Check if the operand is bundled (i.e., a block argument or defined by a
    // Handshake op)
    bool isBundled = isa<BlockArgument>(operand) ||
                     (operand.getDefiningOp() &&
                      operand.getDefiningOp()->getDialect()->getNamespace() ==
                          handshake::HandshakeDialect::getDialectNamespace());
    if (isBundled) {
      // Split the bundled operand
      auto split = splitBundledValue(operand, op->getLoc(), rewriter);
      flatOperands.append(split.getResults().begin(), split.getResults().end());
    } else {
      flatOperands.push_back(operand);
    }
  }

  // --- 3. Create the instance and re-bundle its flat results ---
  hw::InstanceOp hwInstOp = rewriter.create<hw::InstanceOp>(
      op->getLoc(), hwModule,
      StringAttr::get(ctx, uniqueOpName.str() + "_inst"), flatOperands);
  SmallVector<Value> bundledResults;
  for (auto &cast :
       rebundleFlatValues(op->getResultTypes(),
                          SmallVector<Value>(hwInstOp->getResults().begin(),
                                             hwInstOp->getResults().end()),
                          op->getLoc(), rewriter)) {
    assert(cast.getNumResults() == 1);
    bundledResults.push_back(cast.getResult(0));
  }
  assert(bundledResults.size() == op->getNumResults());

  //  Replace all uses of the original operation with the new hw module
  rewriter.replaceOp(op, bundledResults);

  return hwModule;
}

// ---------------------------------------------------------------------------
// OpConversionPattern wrappers
// ---------------------------------------------------------------------------

/// Generic pattern: wraps convertOpToHWModule for any Handshake op T.
template <typename T>
struct ConvertToHWMod : public OpConversionPattern<T> {
  using OpConversionPattern<T>::OpConversionPattern;
  using OpAdaptor = typename T::Adaptor;

  LogicalResult
  matchAndRewrite(T op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return convertOpToHWModule(op, rewriter) ? success() : failure();
  }
};

/// Specialised pattern for handshake::FuncOp.
struct ConvertFuncToHWMod : public OpConversionPattern<handshake::FuncOp> {
  using OpConversionPattern<handshake::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(handshake::FuncOp op, handshake::FuncOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return convertFuncOpToHWModule(op, rewriter) ? success() : failure();
  }
};

//===----------------------------------------------------------------------===//
// Layer 4 — Orchestrator
//
// unbundleAllHandshakeTypes() is the single entry point called by the pass.
// It contains no conversion logic of its own — it just runs three sequential
// phases and delegates everything else to the layers above.
//===----------------------------------------------------------------------===//

// ---------------------------------------------------------------------------
// Phase 1c helper: unrealized-cast elimination
//
// Defined here rather than in Layer 3 because it is not used by patterns:
// it is a post-processing step driven by the orchestrator.
// ---------------------------------------------------------------------------

/// Removes the pairs of UnrealizedConversionCastOps introduced during
/// unbundling.  The expected pattern is:
///
///   value -> cast1 (bundled -> flat) -> cast2 (flat -> bundled) -> user
///
/// cast1 and cast2 cancel out: each use of cast2's result is replaced with
/// the corresponding operand of cast1, then both casts are erased.
LogicalResult removeUnrealizedConversionCasts(mlir::ModuleOp modOp) {
  SmallVector<UnrealizedConversionCastOp> castsToErase;
  // Walk through all unrealized conversion casts in the module
  modOp.walk([&](UnrealizedConversionCastOp castOp1) {
    // Check if the input of the cast 1 is another unrealized
    // conversion cast. If yes, skip it since it does not match the expected
    // pattern.
    if (castOp1.getOperand(0).getDefiningOp<UnrealizedConversionCastOp>()) {
      // Assert that it is not followed by any other cast since a chain of 3
      // casts is unexpected.
      assert(llvm::none_of(castOp1->getUsers(),
                           [](Operation *user) {
                             return isa<UnrealizedConversionCastOp>(user);
                           }) &&
             "unrealized conversion cast removal failed due to chained casts");
      return;
    }
    // Gather the inner casts that consume cast1's results.
    bool allUsersAreCasts = true;
    SmallVector<UnrealizedConversionCastOp> innerCasts;
    for (auto result : castOp1.getResults()) {
      for (auto &use : result.getUses()) {
        if (!isa<UnrealizedConversionCastOp>(use.getOwner())) {
          allUsersAreCasts = false;
        } else {
          innerCasts.push_back(
              cast<UnrealizedConversionCastOp>(use.getOwner()));
        }
      }
      if (!allUsersAreCasts)
        break;
    }
    if (!allUsersAreCasts) {
      // This breaks assumption that casts are chained in pairs
      castOp1.emitError()
          << "unrealized conversion cast removal failed due to complex "
             "usage pattern";
      return;
    }
    // Bypass each inner cast: redirect its result uses to cast1's operands.
    for (auto castOp2 : innerCasts) {
      if (castOp1->getNumOperands() != castOp2->getNumResults()) {
        castOp1.emitError()
            << "unrealized conversion cast removal failed due to "
               "mismatched number of operands and results";
        return;
      }
      for (auto [idx, result] : llvm::enumerate(castOp2.getResults())) {
        result.replaceAllUsesWith(castOp1->getOperand(idx));
      }
      // Add the cast to the list of casts to remove
      castsToErase.push_back(castOp2);
    }
    // Add the cast to the list of casts to remove
    castsToErase.push_back(castOp1);
  });
  // Erase all collected casts
  for (auto castOp : castsToErase) {
    castOp.erase();
  }

  // Check that there are no more unrealized conversion casts
  bool hasCasts = false;
  modOp.walk([&](UnrealizedConversionCastOp castOp) {
    hasCasts = true;
    llvm::errs() << "Remaining unrealized conversion cast: " << castOp << "\n";
  });

  if (hasCasts) {
    modOp.emitError()
        << "unrealized conversion cast removal failed due to remaining "
           "casts";
    return failure();
  }
  return success();
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Converts all Handshake ops in \p modOp into hw::HWModuleOps with flat
/// i1/iN ports, then eliminates the temporary cast ops.
///
/// Three sequential phases:
///   Phase 1a: convert all inner ops (everything except FuncOp / EndOp)
///   Phase 1b: convert the FuncOp (and its inlined EndOp)
///   Phase 1c: remove all UnrealizedConversionCastOps
///
/// Phases 1a and 1b are separate because FuncOp must be converted after all
/// its child ops: the child conversions create hw instances that are placed
/// inside the hw module that replaces FuncOp.
LogicalResult unbundleAllHandshakeTypes(ModuleOp modOp, MLIRContext *ctx) {

  // ---- Phase 1a: convert all inner Handshake ops --------------------------
  RewritePatternSet patterns(ctx);
  ChannelUnbundlingTypeConverter typeConverter;
  ConversionTarget target(*ctx);
  target.addLegalDialect<synth::SynthDialect>();
  target.addLegalDialect<hw::HWDialect>();
  // Add casting as legal
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.addIllegalDialect<handshake::HandshakeDialect>();
  // FuncOp and EndOp are handled in Phase 1b, so we mark them as legal here
  target.addLegalOp<handshake::FuncOp>();
  target.addLegalOp<handshake::EndOp>();
  patterns.insert<
      ConvertToHWMod<handshake::BufferOp>, ConvertToHWMod<handshake::NDWireOp>,
      ConvertToHWMod<handshake::ConditionalBranchOp>,
      ConvertToHWMod<handshake::BranchOp>, ConvertToHWMod<handshake::MergeOp>,
      ConvertToHWMod<handshake::ControlMergeOp>,
      ConvertToHWMod<handshake::MuxOp>, ConvertToHWMod<handshake::JoinOp>,
      ConvertToHWMod<handshake::BlockerOp>, ConvertToHWMod<handshake::SourceOp>,
      ConvertToHWMod<handshake::ConstantOp>, ConvertToHWMod<handshake::SinkOp>,
      ConvertToHWMod<handshake::ForkOp>, ConvertToHWMod<handshake::LazyForkOp>,
      ConvertToHWMod<handshake::LoadOp>, ConvertToHWMod<handshake::StoreOp>,
      ConvertToHWMod<handshake::ReadyRemoverOp>,
      ConvertToHWMod<handshake::ValidMergerOp>,
      ConvertToHWMod<handshake::SharingWrapperOp>,
      ConvertToHWMod<handshake::MemoryControllerOp>,

      // Arith operations
      ConvertToHWMod<handshake::AddFOp>, ConvertToHWMod<handshake::AddIOp>,
      ConvertToHWMod<handshake::AndIOp>, ConvertToHWMod<handshake::CmpFOp>,
      ConvertToHWMod<handshake::CmpIOp>, ConvertToHWMod<handshake::DivFOp>,
      ConvertToHWMod<handshake::DivSIOp>, ConvertToHWMod<handshake::DivUIOp>,
      ConvertToHWMod<handshake::RemSIOp>, ConvertToHWMod<handshake::ExtSIOp>,
      ConvertToHWMod<handshake::ExtUIOp>, ConvertToHWMod<handshake::MulFOp>,
      ConvertToHWMod<handshake::MulIOp>, ConvertToHWMod<handshake::NegFOp>,
      ConvertToHWMod<handshake::OrIOp>, ConvertToHWMod<handshake::SelectOp>,
      ConvertToHWMod<handshake::ShLIOp>, ConvertToHWMod<handshake::ShRSIOp>,
      ConvertToHWMod<handshake::ShRUIOp>, ConvertToHWMod<handshake::SubFOp>,
      ConvertToHWMod<handshake::SubIOp>, ConvertToHWMod<handshake::TruncIOp>,
      ConvertToHWMod<handshake::TruncFOp>, ConvertToHWMod<handshake::XOrIOp>,
      ConvertToHWMod<handshake::SIToFPOp>, ConvertToHWMod<handshake::UIToFPOp>,
      ConvertToHWMod<handshake::FPToSIOp>, ConvertToHWMod<handshake::ExtFOp>,
      ConvertToHWMod<handshake::AbsFOp>, ConvertToHWMod<handshake::MaxSIOp>,
      ConvertToHWMod<handshake::MaxUIOp>, ConvertToHWMod<handshake::MinSIOp>,
      ConvertToHWMod<handshake::MinUIOp>,

      // Speculative operations
      ConvertToHWMod<handshake::SpecCommitOp>,
      ConvertToHWMod<handshake::SpecSaveOp>,
      ConvertToHWMod<handshake::SpecSaveCommitOp>,
      ConvertToHWMod<handshake::SpeculatorOp>,
      ConvertToHWMod<handshake::SpeculatingBranchOp>,
      ConvertToHWMod<handshake::NonSpecOp>>(typeConverter, ctx);
  if (failed(applyPartialConversion(modOp, target, std::move(patterns))))
    return failure();

  // ---- Phase 1b: convert the top-level FuncOp ----------------------------
  RewritePatternSet funcPatterns(ctx);
  ConversionTarget funcTarget(*ctx);
  funcTarget.addLegalDialect<synth::SynthDialect>();
  funcTarget.addLegalDialect<hw::HWDialect>();
  // Add casting as legal
  funcTarget.addLegalOp<UnrealizedConversionCastOp>();
  funcTarget.addIllegalDialect<handshake::HandshakeDialect>();
  funcPatterns.insert<ConvertFuncToHWMod>(typeConverter, ctx);
  if (failed(
          applyPartialConversion(modOp, funcTarget, std::move(funcPatterns))))
    return failure();

  // ---- Phase 1c: remove all unrealized conversion casts ------------------
  if (failed(removeUnrealizedConversionCasts(modOp)))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// Step 2 — SignalRewriter implementation
//===----------------------------------------------------------------------===//

// Utility function to get the string formatted in the desired portname[idx]
// format
// If the root name already contains an index, the old index is linearized and
// added to the new one using the width input
std::string formatPortName(const std::string &rootName,
                           const std::optional<unsigned> &index,
                           unsigned width = 0) {
  if (index.has_value()) {
    unsigned newIndex = index.value();
    std::string newRootName = rootName;
    // Check if there is already present an index
    std::regex pattern(R"((\w+)\[(\d+)\])");
    // We use regex to identify this pattern
    std::smatch matches;
    if (std::regex_match(rootName, matches, pattern)) {
      // If the pattern matches, assert the width is not 0
      assert(width != 0 &&
             "the width of the signal cannot be 0 if it is an array");
      // Linearize old index
      std::string oldIndex = matches[2].str();
      unsigned oldIndexLinearized = std::stoi(oldIndex) * width;
      // Add it to new one
      newIndex += oldIndexLinearized;
      newRootName = matches[1].str();
    }
    return newRootName + "[" + std::to_string(newIndex) + "]";
  }
  return rootName;
}

// Function to get a new module name for the rewritten hw module
mlir::StringAttr
SignalRewriter::getRewrittenModuleName(hw::HWModuleOp oldMod,
                                       mlir::MLIRContext *ctx) {
  return mlir::StringAttr::get(ctx, oldMod.getName() + "_rewritten");
}

// Function to get the mapping of an input signal from the old module to the
// new module
SmallVector<Value> SignalRewriter::getInputSignalMapping(Value oldInputSignal,
                                                         OpBuilder builder,
                                                         Location loc) {
  auto it = oldModuleSignalToNewModuleSignalsMap.find(oldInputSignal);
  if (it != oldModuleSignalToNewModuleSignalsMap.end()) {
    return it->second;
  }
  // If there is no mapping, it means that the value is not yet
  // available since the instance that produces it has not been
  // rewritten yet. In this case, check if we already created a
  // temporary value for this ready signal
  auto tempIt = oldModuleSignalToTempValuesMap.find(oldInputSignal);
  if (tempIt != oldModuleSignalToTempValuesMap.end()) {
    return tempIt->second;
  }
  // Else, create a new temporary hw constant to hold the connection for each
  // bit
  SmallVector<Value> tempValues;
  Type signalType = oldInputSignal.getType();
  assert(signalType.isa<IntegerType>() &&
         "only integer types are supported for signals for now");
  unsigned bitWidth = signalType.cast<IntegerType>().getWidth();
  assert(bitWidth > 0 && "signal must have positive bit width");
  for (unsigned i = 0; i < bitWidth; ++i) {
    auto tempConst = builder.create<hw::ConstantOp>(
        loc, oldInputSignal.getType(),
        builder.getIntegerAttr(oldInputSignal.getType(), 0));
    tempValues.push_back(tempConst.getResult());
  }
  // Store the temporary value in the map
  oldModuleSignalToTempValuesMap[oldInputSignal] = tempValues;
  // Use the result of the constant as the operand
  return tempValues;
}

// Function to update the mapping between old module signals and new module
// signals after getting a new result value
void SignalRewriter::updateOutputSignalMapping(
    Value oldResult, StringRef outputName, int oldOutputIdx,
    hw::HWModuleOp oldMod, hw::HWModuleOp newMod, hw::InstanceOp newInst) {
  // Check if you can find the output idx of the oldResult in the list
  // oldOutputIdxToNewOutputIdxMap
  OpBuilder builder(oldMod);
  StringAttr oldModName = builder.getStringAttr(oldMod.getName());
  SmallVector<std::pair<unsigned, SmallVector<unsigned>>>
      oldOutIdxToNewOutIdxs = oldOutputIdxToNewOutputIdxMap[oldModName];
  SmallVector<unsigned> outputIdxsNewInst = {};
  for (auto &pair : oldOutIdxToNewOutIdxs) {
    if (oldOutputIdx != -1 &&
        pair.first == static_cast<unsigned>(oldOutputIdx)) {
      outputIdxsNewInst = pair.second;
      break;
    }
  }
  // If you cannot find it, use the output name to find it
  if (outputIdxsNewInst.empty()) {
    // The only case in which this is possible is when it is a ready signal
    assert(outputName.contains("_ready") &&
           "could not find output index mapping for non-ready signal");
    // Find the corresponding output index of the output in the new module
    for (auto &p : newMod.getPortList()) {
      StringRef portName = p.name.getValue();
      if (portName == outputName) {
        outputIdxsNewInst.push_back(p.argNum);
      }
    }
  }
  assert(!outputIdxsNewInst.empty() &&
         "could not find output port in new module");
  SmallVector<Value> newResults;
  for (unsigned idx : outputIdxsNewInst) {
    newResults.push_back(newInst->getResult(idx));
  }
  // Add mapping between old non-ready signal and new non-ready signal
  oldModuleSignalToNewModuleSignalsMap[oldResult] = newResults;
  // Check if any temporary value has been created for the new result
  // value by seeing if the old output value is in the map of temporary
  // values
  auto tempIt = oldModuleSignalToTempValuesMap.find(oldResult);
  if (tempIt != oldModuleSignalToTempValuesMap.end()) {
    assert(newResults.size() == tempIt->second.size() &&
           "mismatched number of bits between temporary value and new result");
    for (size_t i = 0; i < newResults.size(); ++i) {
      // Replace all uses of the temporary value with the new result value
      tempIt->second[i].replaceAllUsesWith(newResults[i]);
    }
    for (auto tempVal : tempIt->second) {
      // Remove the operation creating the temporary value
      tempVal.getDefiningOp()->erase();
    }
    // Remove the temporary value from the map
    oldModuleSignalToTempValuesMap.erase(tempIt);
  }
}

/// Rewrite an HW instance to use the rewritten module interface and operands.
/// This updates operand connections, reconstructs result groups, and updates
/// internal mapping structures used by the module-level rewriting.
void SignalRewriter::rewriteHWInstance(
    hw::InstanceOp oldInst, ModuleOp parent, SymbolTable &symTable,
    DenseMap<StringRef, hw::HWModuleOp> &newHWmodules,
    DenseMap<StringRef, hw::HWModuleOp> &oldHWmodules) {

  // This function executes the following steps:
  // 1. Check if the instance's module has already been rewritten. If not,
  //    rewrite it.
  // 2. Create a new instance operation with the new module and updated
  //    operands.
  // 3. Update the mapping between old signals and new signals for the outputs
  //    of the new hw instance.

  // Step 1: Check if the instance's module has already been rewritten
  OpBuilder builder(parent);
  StringRef moduleName = oldInst.getModuleName();
  // If it has not been processed yet, rewrite it
  if (!newHWmodules.count(moduleName)) {
    hw::HWModuleOp oldMod = symTable.lookup<hw::HWModuleOp>(moduleName);
    if (oldMod) {
      rewriteHWModule(oldMod, parent, symTable, newHWmodules, oldHWmodules);
    }
  }

  // Step 2: Create a new instance operation with the new module and updated
  // operands.
  // Get the new hw module after rewriting to be used for the new instance
  hw::HWModuleOp newMod = newHWmodules[moduleName];
  assert(newMod && "could not find new hw module for instance");

  // Get the old hw module to identify the ready signals to change
  hw::HWModuleOp oldMod = oldHWmodules[moduleName];
  assert(oldMod && "could not find old hw module for instance");

  // Get the top-level module of the old instance
  hw::HWModuleOp oldInstTopModule = oldInst->getParentOfType<hw::HWModuleOp>();
  StringRef oldInstTopModuleName = oldInstTopModule.getName();
  // Find the corresponding new module in the new hw modules map where to
  // insert the new operations
  hw::HWModuleOp newTopMod = newHWmodules[oldInstTopModuleName];
  assert(newTopMod && "could not find new top module for instance");
  // Create the new operations within the new hw module
  builder.setInsertionPoint(newTopMod.getBodyBlock()->getTerminator());
  Location locNewOps = newTopMod.getBodyBlock()->getTerminator()->getLoc();

  // Save the list of outputs that are not ready signals to replace uses later
  SmallVector<std::pair<StringRef, unsigned>> nonReadyOutputs;
  // Save the list of the old ready inputs which will be mapped to the new
  // ready output values
  SmallVector<std::pair<StringRef, Value>> oldReadyInputs;
  // Save the new operands for the new instance
  SmallVector<Value> newOperands;
  // Iterate through the name of the ports to identify the mapping between
  // signals of the old hw module and new hw module
  for (auto &port : oldMod.getPortList()) {
    bool isReady = port.name.getValue().contains("ready");
    if (port.isInput() && isReady) {
      // This is an input of the old module and should become an output of the
      // new module. Save the old input value to map it later to the
      // corresponding new output value
      oldReadyInputs.push_back(std::make_pair(port.name.getValue(),
                                              oldInst.getOperand(port.argNum)));
    } else if (port.isOutput() && isReady) {
      // This is an output of the old module and should become an input of the
      // new module. Check if there is a mapping for this ready signal
      Value oldReadyOutput = oldInst->getResult(port.argNum);
      SmallVector<Value> newReadyInputs =
          getInputSignalMapping(oldReadyOutput, builder, locNewOps);
      assert(newReadyInputs.size() == 1 && "ready signal should be 1 bit wide");
      Value newReadyInput = newReadyInputs[0];
      newOperands.push_back(newReadyInput);
    } else if (port.isInput()) {
      // Check if there is a mapping for this non-ready signal
      Value oldNonReadyInput = oldInst.getOperand(port.argNum);
      SmallVector<Value> newNonReadyInputs =
          getInputSignalMapping(oldNonReadyInput, builder, locNewOps);
      // Append each bit separately of the non-ready signal to the new
      // operands
      for (auto newNonReadyInput : newNonReadyInputs)
        newOperands.push_back(newNonReadyInput);
    } else {
      // Collect non-ready outputs to replace uses later
      nonReadyOutputs.push_back(
          std::make_pair(port.name.getValue(), port.argNum));
    }
  }

  // Add new inputs for clk and rst from the top function
  newOperands.push_back(oldModuleSignalToNewModuleSignalsMap[clkSignalTop][0]);
  newOperands.push_back(oldModuleSignalToNewModuleSignalsMap[rstSignalTop][0]);

  // Create the new instance operation within the new hw module
  auto newInst = builder.create<hw::InstanceOp>(
      locNewOps, newMod, oldInst.getInstanceNameAttr(), newOperands);

  // Step 3: Update the mapping between old signals and new signals after
  // getting the new result values
  for (auto [outputName, outputIdxOldInst] : nonReadyOutputs) {
    // Find the output port in the old module
    Value oldResult = oldInst.getResult(outputIdxOldInst);
    updateOutputSignalMapping(oldResult, outputName, outputIdxOldInst, oldMod,
                              newMod, newInst);
  }
  // Update the mapping between old ready inputs and new ready outputs after
  // getting the new result values
  for (auto [inputName, oldReadyInput] : oldReadyInputs) {
    updateOutputSignalMapping(oldReadyInput, inputName, -1, oldMod, newMod,
                              newInst);
  }
}

/// Rewrite a HW module to normalize signal directions (e.g., ready going
/// opposite to data/valid) and unbundle multi-bit signals into single-bit
/// signals according to the chosen convention.
void SignalRewriter::rewriteHWModule(
    hw::HWModuleOp oldMod, ModuleOp parent, SymbolTable &symTable,
    DenseMap<StringRef, hw::HWModuleOp> &newHWmodules,
    DenseMap<StringRef, hw::HWModuleOp> &oldHWmodules) {

  // This function executes the following steps:
  // 1. Check if the module has already been rewritten. If so, return.
  // 2. Create a new hw module with inverted ready signal directions.
  // 3. Iterate through the body operations of the old module:
  //    a. If the operation is an hw instance, rewrite it to fix ready signal
  //       directions.
  //    b. If the operation is not an hw instance, check if it is only
  //       output operations and synth subckt operations. If so, create a
  //       synth subckt operation to connect the module ports. If not, raise
  //       an error.
  // 4. Finally, connect the hw instances to the terminator operands of the
  // new
  //    module.

  // Step 1: Check if the module has already been rewritten
  if (newHWmodules.count(oldMod.getName())) {
    return;
  }

  // Step 2: Create a new hw module with inverted ready signal directions
  MLIRContext *ctx = parent.getContext();
  OpBuilder builder(parent);

  SmallVector<hw::PortInfo> newInputs;
  SmallVector<hw::PortInfo> newOutputs;

  // Collect the new inputs and outputs with inverted ready signal directions
  unsigned inputIdx = 0;
  unsigned outputIdx = 0;
  // Store mapping from new output idx start and end to old signal
  SmallVector<std::pair<std::pair<unsigned, unsigned>, Value>>
      newModuleOutputIdxToOldSignal;
  // Store mapping from new input idx start and end to old signal
  SmallVector<std::pair<std::pair<unsigned, unsigned>, Value>>
      newModuleInputIdxToOldSignal;
  // Store the mapping between old output signal idx and new output signal idxs.
  // This should be applied only to non-ready signals.
  SmallVector<std::pair<unsigned, SmallVector<unsigned>>>
      oldOutputIdxToNewOutputIdxs;
  // Iterate over the ports of the old hw module
  for (auto &p : oldMod.getPortList()) {
    bool isReady = p.name.getValue().contains("ready");

    unsigned startOutputIdx = outputIdx;
    unsigned startInputIdx = inputIdx;
    if ((p.isInput() && isReady) || (p.isOutput() && !isReady)) {
      // If the port is input and ready, it becomes output and ready in the
      // new module. If a port is output and not ready, it stays as such.
      Type signalType = p.type;
      assert(signalType.isa<IntegerType>() &&
             "only integer types are supported for signals for now");
      unsigned bitWidth = signalType.cast<IntegerType>().getWidth();
      for (unsigned i = 0; i < bitWidth; ++i) {
        // Update portname indexing the specific bit unless it is 1 bit wide
        std::string portName =
            formatPortName(p.name.getValue().str(), i, bitWidth);
        if (bitWidth == 1) {
          portName = p.name.getValue().str();
        }
        newOutputs.push_back({hw::ModulePort{StringAttr::get(ctx, portName),
                                             builder.getIntegerType(1),
                                             hw::ModulePort::Direction::Output},
                              outputIdx});
        outputIdx++;
      }
      Value oldSignal;
      if (isReady) {
        // Record mapping from new ready output to old ready input
        oldSignal = oldMod.getBodyBlock()->getArgument(p.argNum);
      } else {
        // Record mapping from new non-ready output to old non-ready output
        oldSignal =
            oldMod.getBodyBlock()->getTerminator()->getOperand(p.argNum);
      }
      unsigned endOutputIdx = outputIdx - 1;
      newModuleOutputIdxToOldSignal.push_back(std::make_pair(
          std::make_pair(startOutputIdx, endOutputIdx), oldSignal));
      if (!isReady) {
        // Store mapping from old output idx to new output idxs for
        // non-ready outputs
        SmallVector<unsigned> newOutputIdxs;
        for (unsigned i = startOutputIdx; i <= endOutputIdx; ++i) {
          newOutputIdxs.push_back(i);
        }
        oldOutputIdxToNewOutputIdxs.push_back(
            std::make_pair(p.argNum, newOutputIdxs));
      }
    } else if ((p.isOutput() && isReady) || (p.isInput() && !isReady)) {
      // If the port is output and ready, it becomes input and ready in the
      // new module. If a port is input and not ready, it stays as such.
      Type signalType = p.type;
      assert(signalType.isa<IntegerType>() &&
             "only integer types are supported for signals for now");
      unsigned bitWidth = signalType.cast<IntegerType>().getWidth();
      for (unsigned i = 0; i < bitWidth; ++i) {
        // Update portname indexing the specific bit unless it is 1 bit wide
        std::string portName =
            formatPortName(p.name.getValue().str(), i, bitWidth);
        if (bitWidth == 1) {
          portName = p.name.getValue().str();
        }
        newInputs.push_back(
            {hw::ModulePort{StringAttr::get(ctx, StringRef{portName}),
                            builder.getIntegerType(1),
                            hw::ModulePort::Direction::Input},
             inputIdx});
        inputIdx++;
      }
      Value oldSignal;
      if (isReady) {
        // Record mapping from new ready input to old ready output
        oldSignal =
            oldMod.getBodyBlock()->getTerminator()->getOperand(p.argNum);
      } else {
        // Record mapping from new non-ready input to old non-ready input
        oldSignal = oldMod.getBodyBlock()->getArgument(p.argNum);
      }
      unsigned endInputIdx = inputIdx - 1;
      newModuleInputIdxToOldSignal.push_back(std::make_pair(
          std::make_pair(startInputIdx, endInputIdx), oldSignal));
    } else {
      assert(false && "port is neither input nor output");
    }
  }

  // Record the old output idx to new output idxs mapping for non-ready outputs
  oldOutputIdxToNewOutputIdxMap[builder.getStringAttr(oldMod.getName())] =
      oldOutputIdxToNewOutputIdxs;

  // Add clk and rst for all new inputs of hw modules
  newInputs.push_back({hw::ModulePort{mlir::StringAttr::get(ctx, clockSignal),
                                      builder.getIntegerType(1),
                                      hw::ModulePort::Direction::Input},
                       inputIdx});
  unsigned clkInputIdx = inputIdx;
  inputIdx++;
  newInputs.push_back({hw::ModulePort{mlir::StringAttr::get(ctx, resetSignal),
                                      builder.getIntegerType(1),
                                      hw::ModulePort::Direction::Input},
                       inputIdx});
  unsigned rstInputIdx = inputIdx;
  inputIdx++;

  hw::ModulePortInfo newPortInfo(newInputs, newOutputs);

  // Create new hw module
  builder.setInsertionPointAfter(oldMod);
  mlir::StringAttr newModuleName = getRewrittenModuleName(oldMod, ctx);
  auto newMod = builder.create<hw::HWModuleOp>(oldMod.getLoc(), newModuleName,
                                               newPortInfo);
  // Save it in the list of new module using the same name as the old module
  // as key
  newHWmodules[oldMod.getName()] = newMod;

  // Copy the blif path attribute from the old module to the new module
  if (opToBlifPathMap.contains(oldMod.getOperation())) {
    auto blifPath = opToBlifPathMap[oldMod.getOperation()];
    opToBlifPathMap[newMod.getOperation()] = blifPath;
    // Remove old module from the map
    opToBlifPathMap.erase(oldMod.getOperation());
  }

  // Add mapping between new inputs to old signals
  for (auto [newModuleInputIdxs, oldSignal] : newModuleInputIdxToOldSignal) {
    unsigned newModuleInputIdxStart = newModuleInputIdxs.first;
    unsigned newModuleInputIdxEnd = newModuleInputIdxs.second;
    // For each bit of the input signal, get the corresponding argument
    SmallVector<Value> newReadyInput;
    for (unsigned idx = newModuleInputIdxStart; idx <= newModuleInputIdxEnd;
         ++idx) {
      newReadyInput.push_back(newMod.getBodyBlock()->getArgument(idx));
    }
    assert(newReadyInput.size() && "could not find mapping for input signal");
    oldModuleSignalToNewModuleSignalsMap[oldSignal] = newReadyInput;
  }

  // If the hw module represent the top function
  if (oldMod.getName() == getTopFunctionName()) {
    // Save the clock and reset signals
    assert((clkSignalTop == nullptr && rstSignalTop == nullptr) &&
           "reset and clock signals should be set only once for the top "
           "function");
    clkSignalTop = newMod.getBodyBlock()->getArgument(clkInputIdx);
    rstSignalTop = newMod.getBodyBlock()->getArgument(rstInputIdx);
    oldModuleSignalToNewModuleSignalsMap[clkSignalTop] = {clkSignalTop};
    oldModuleSignalToNewModuleSignalsMap[rstSignalTop] = {rstSignalTop};
  }

  // Step 3: Iterate through the body operations of the old module
  bool hasHwInstances = true;
  // Iterate through old body operations and invert ready signals in instances
  for (auto &op : oldMod.getBody().getOps()) {
    // Check if the operation is an hw instance
    if (auto instOp = dyn_cast<hw::InstanceOp>(op)) {
      // Step 3a: If the operation is an hw instance, rewrite it to fix ready
      // signal directions.
      rewriteHWInstance(instOp, parent, symTable, newHWmodules, oldHWmodules);
    } else if (!isa<hw::OutputOp>(op)) {
      hasHwInstances = false;
    }
  }

  if (!hasHwInstances) {
    // Step 3b: If the operation is not an hw instance, check if it is only
    // output operations and synth subckt operations. If so, create a
    // synth subckt operation to connect the module ports. If not, raise an
    // error.
    for (auto &op : oldMod.getBody().getOps()) {
      if (!isa<hw::OutputOp>(op) && !isa<synth::SubcktOp>(op)) {
        llvm::errs() << "Found non-instance operation in hw module: " << op
                     << "\n";
        assert(false && "found non-instance operation in hw module");
      }
    }
    ConversionPatternRewriter rewriter(ctx);
    if (failed(instantiateSynthPlaceholder(newMod, rewriter))) {
      assert(false && "synth instantiation in hw module failed");
    }
    return;
  }

  // Step 4: Finally, connect the hw instances to the terminator operands of
  // the new module.
  builder.setInsertionPointToEnd(newMod.getBodyBlock());
  SmallVector<Value> newTerminatorOperands;
  for (auto [newOutputIdxs, oldSignal] : newModuleOutputIdxToOldSignal) {
    SmallVector<Value> newModuleOutputs =
        oldModuleSignalToNewModuleSignalsMap[oldSignal];
    assert(newModuleOutputs.size() &&
           "could not find mapping for output signal");
    newTerminatorOperands.append(newModuleOutputs.begin(),
                                 newModuleOutputs.end());
  }

  // Add operands to existing terminator
  Operation *newTerminator = newMod.getBodyBlock()->getTerminator();
  newTerminator->setOperands(newTerminatorOperands);
}

/// Function to apply signal-direction and bit-level rewrites to all HW
/// modules in the MLIR module.
LogicalResult SignalRewriter::rewriteAllSignals(mlir::ModuleOp modOp) {

  // The following function iterates through all hw modules in the module
  // and rewrites them to invert the direction of ready signals to follow
  // the standard handshake protocol where ready signals go in the opposite
  // direction with respect to data and valid signals. It applies recursively
  // to all hw modules instantiated within other hw modules.
  // It does so by creating new hw modules with the rewritten ready signal
  // directions and then connecting them following the same graph structure of
  // the old modules. Finally, it removes the old hw modules and renames the
  // new hw modules to the original names. Additionally, during the ready
  // signal inversion, it also unbundles multi-bit signals into single-bit
  // signals.

  // Maps to keep track of old and new hw modules
  // The old hw modules have the wrong ready signal directions where ready
  // signals follows the same direction as data and valid signals
  // The new hw modules will contain the rewritten modules with correct ready
  // signal directions
  DenseMap<StringRef, hw::HWModuleOp> oldHWModules;
  DenseMap<StringRef, hw::HWModuleOp> newHWModules;
  // Get symbol table for hw modules
  SymbolTable symTable(modOp);
  // Collect all hw modules in the module
  modOp.walk([&](hw::HWModuleOp m) { oldHWModules.insert({m.getName(), m}); });
  // Iterate through all hw modules and rewrite them
  for (auto [name, hwMod] : oldHWModules)
    rewriteHWModule(hwMod, modOp, symTable, newHWModules, oldHWModules);
  // Erase old hw modules
  for (auto [modName, oldMod] : oldHWModules) {
    oldMod.erase();
  }
  // Iterate through all new hw modules and rename them to the original names
  for (auto [originalName, newMod] : newHWModules) {
    mlir::StringAttr originalNameAttr =
        mlir::StringAttr::get(modOp.getContext(), originalName);
    StringRef currentModName = newMod.getName();
    // Change the module name of the instances of this module
    modOp.walk([&](hw::InstanceOp instOp) {
      if (instOp.getModuleName() == currentModName) {
        instOp.setModuleName(originalNameAttr);
      }
    });
    // Rename the new module to the original name
    newMod.setName(originalName);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Step 3 — Populate hw modules with BLIF-imported netlists
//===----------------------------------------------------------------------===//

/// Replaces the synth::SubcktOp placeholder body of the hw module referenced
/// by \p instOp with the gate-level netlist imported from its BLIF file.
/// If the module has already been populated (its name is in
/// \p alreadyPopulatedHWMod) the function returns immediately.
LogicalResult
populateHWModuleWithSynthOps(ModuleOp modOp, hw::InstanceOp op,
                             SmallVector<std::string> &alreadyPopulatedHWMod) {

  // First, this function imports the blif circuit as a new hw module. Then, it
  // replaces the body of the original hw module with the body of the new hw
  // module.

  // Check if the hw module of the instance has already been modified to not
  // modify it multiple times
  if (std::find(alreadyPopulatedHWMod.begin(), alreadyPopulatedHWMod.end(),
                op.getModuleName().str()) != alreadyPopulatedHWMod.end()) {
    return success();
  }

  // Ensure that the blif path is specified in the hw module of the hw
  // instance operation
  SymbolTable symTable = SymbolTable(op->getParentOfType<ModuleOp>());
  hw::HWModuleOp hwModule = symTable.lookup<hw::HWModuleOp>(op.getModuleName());
  if (!hwModule) {
    llvm::errs() << "could not find hw module for instance: " << op << "\n";
    return failure();
  }
  // Get the blif path
  assert(opToBlifPathMap.contains(hwModule.getOperation()) &&
         "missing blif path for hw module");
  StringRef blifFilePath = opToBlifPathMap[hwModule.getOperation()];

  // Check if blifFilePath is empty
  if (blifFilePath == "") {
    // If the blif path is empty, it means that this module should not be
    // replaced with synth operations, so just return success without doing
    // anything
    return success();
  }

  // Collect the pins of the old hw module to ensure the new hw module has the
  // same pins ordering
  SmallVector<std::string> oldInputPins;
  SmallVector<std::string> oldOutputPins;
  for (auto &port : hwModule.getPortList()) {
    if (port.isInput()) {
      oldInputPins.push_back(port.name.getValue().str());
    } else {
      oldOutputPins.push_back(port.name.getValue().str());
    }
  }
  std::pair<SmallVector<std::string>, SmallVector<std::string>> oldPins = {
      oldInputPins, oldOutputPins};

  // Import the blif circuit corresponding to the hw module of the instance
  hw::HWModuleOp newHWModule = importBlifCircuit(modOp, blifFilePath, oldPins);

  // Check if the import was successful
  if (!newHWModule) {
    llvm::errs() << "failed to import blif circuit for instance: " << op
                 << "\n";
    return failure();
  }

  // Replace the hw module with the new one
  std::string originalModuleName = hwModule.getName().str();
  std::string newModuleName = newHWModule.getName().str();
  // Replace the name of the new module with the name of the original module
  newHWModule.setName(originalModuleName);
  // Remove from symbol table the original module and add the new module with
  // the original name
  symTable.erase(hwModule);
  symTable.insert(newHWModule);
  // Add the name of the modified module to the list to avoid modifying it
  alreadyPopulatedHWMod.push_back(originalModuleName);

  return success();
}

// Function to import the blif circuits corresponding to the original
// handshake units
LogicalResult populateHWModules(mlir::ModuleOp modOp, StringRef topModuleName,
                                MLIRContext *ctx) {
  // The following function iterates through all the hw modules and populate
  // them with the synth operations like registers, combinational logic, etc.
  // if possible. The description of the implementation is defined in the path
  // specified by the attribute blifPathAttrStr on each hw module

  // Get hw module corresponding to the top module
  SymbolTable symTable(modOp);
  hw::HWModuleOp topHWModule = symTable.lookup<hw::HWModuleOp>(topModuleName);
  if (!topHWModule) {
    llvm::errs() << "could not find hw module for top module: " << topModuleName
                 << "\n";
    return failure();
  }

  // Collect all hw instances in the top module
  SmallVector<hw::InstanceOp> hwInstances;
  topHWModule.walk([&](hw::InstanceOp op) { hwInstances.push_back(op); });
  OpBuilder builder(ctx);
  // Collect the list of modified hw modules to avoid modifying the same
  // module multiple times
  SmallVector<std::string> alreadyPopulatedHWMod;
  // Iterate through each hw instance and populate the corresponding hw module
  // with synth operations
  for (hw::InstanceOp hwInst : hwInstances) {
    if (failed(populateHWModuleWithSynthOps(modOp, hwInst,
                                            alreadyPopulatedHWMod))) {
      llvm::errs() << "Failed to convert hw instance to synth ops: " << hwInst
                   << "\n";
      return failure();
    }
  }
  return success();
}

// ------------------------------------------------------------------
// Main pass definition
// ------------------------------------------------------------------

namespace {
// The following pass converts handshake operations into synth operations
// It executes in multiple steps:
// 0) Check each handshake operation is marked with the path of the blif file
//    where its definition is located. This is done in a separate pass
//    (--mark-handshake-blif-impl).
// 1) Unbundle all handshake types used in the handshake function
// 2) Invert the direction of ready signals in all hw modules and hw instances
//    to follow the standard handshake protocol where ready signals go in the
//    opposite direction with respect to data and valid signals. Additionally,
//    data signals are unbundled into single-bit signals.
// 3) Populate the hw module operations with the correspoding synth operations.
//    The description of the implementation is defined in the path specified
//    by the attribute blifPathAttrStr on each hw module
class HandshakeToSynthPass
    : public dynamatic::impl::HandshakeToSynthBase<HandshakeToSynthPass> {
public:
  HandshakeToSynthPass(std::string blifDirPath);
  using HandshakeToSynthBase::HandshakeToSynthBase;

  void runDynamaticPass() override {
    mlir::ModuleOp modOp = getOperation();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);
    // We only support one function per module
    handshake::FuncOp funcOp = nullptr;
    // Check this is the case
    for (auto op : modOp.getOps<handshake::FuncOp>()) {
      if (op.isExternal())
        continue;
      if (funcOp) {
        modOp->emitOpError() << "we currently only support one non-external "
                                "handshake function per module";
        return signalPassFailure();
      }
      funcOp = op;
    }
    // If there is no function, nothing to do
    if (!funcOp)
      return;
    // Save the name of the top module function
    StringRef topModuleName = funcOp.getName();

    // Step 0: Check each handshake operation is marked with the path of the
    // blif file
    funcOp->walk([&](Operation *op) {
      if (isa<handshake::FuncOp>(op))
        return;
      BLIFImplInterface blifImplInterface = dyn_cast<BLIFImplInterface>(op);
      if (!blifImplInterface) {
        llvm::errs() << "Handshake operation " << getUniqueName(op)
                     << " does not implement the BLIFImplInterface\n";
        assert(false && "Check the pass that marks handshake operations with "
                        "BLIF file paths is executed correctly");
      }
      std::string blifFilePath = blifImplInterface.getBLIFImpl().str();
      if (blifFilePath.empty()) {
        // Write out warning
        llvm::errs() << "Warning: Handshake operation " << getUniqueName(op)
                     << " has an empty BLIF file path\n";
      }
    });

    // Step 1: unbundle all handshake types in the handshake operations
    if (failed(unbundleAllHandshakeTypes(modOp, ctx)))
      return signalPassFailure();

    // Step 2: invert the direction of all ready signals in the hw modules
    // created from handshake operations. Additionally, unbundle data signals
    // into single-bit signals.
    //
    // Create on object of the SignalRewriter class to manage the
    // inversion
    SignalRewriter signalRewriter;
    // Set the name of the top function
    signalRewriter.setTopFunctionName(topModuleName);
    if (failed(signalRewriter.rewriteAllSignals(modOp)))
      return signalPassFailure();

    // Step 3: Populate the hw module operations with the correspoding synth
    // operations. The description of the implementation is defined in the
    // path specified by the attribute blifPathAttrStr on each hw module.
    if (failed(populateHWModules(modOp, topModuleName, ctx)))
      return signalPassFailure();
  }
};

} // namespace