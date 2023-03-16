//===- HandshakeToNetlist.cpp - Converts handshake to HW/ESI ----*- C++ -*-===//
//
// This file contains the implementation of the handshake to netlist pass.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Conversion/HandshakeToNetlist.h"
#include "circt/Conversion/HandshakeToHW.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "circt/Support/BackedgeBuilder.h"
#include "dynamatic/Conversion/PassDetails.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace circt;
using namespace circt::handshake;
using namespace circt::hw;
using namespace dynamatic;

using NameUniquer = std::function<std::string(Operation *)>;

namespace {

// Shared state used by various functions; captured in a struct to reduce the
// number of arguments that we have to pass around.
struct HandshakeLoweringState {
  ModuleOp parentModule;
  NameUniquer nameUniquer;
  OpBuilder extModBuilder;

  HandshakeLoweringState(ModuleOp mod, NameUniquer nameUniquer)
      : parentModule(mod), nameUniquer(nameUniquer),
        extModBuilder(mod->getContext()) {
    // Set insertion point to start of module
    extModBuilder.setInsertionPointToStart(mod.getBody());
  }
};

// A type converter is needed to perform the in-flight materialization of "raw"
// (non-ESI channel) types to their ESI channel correspondents. This comes into
// effect when backedges exist in the input IR.
class ESITypeConverter : public TypeConverter {
public:
  ESITypeConverter() {
    addConversion([](Type type) -> Type { return esiWrapper(type); });

    addTargetMaterialization(
        [&](mlir::OpBuilder &builder, mlir::Type resultType,
            mlir::ValueRange inputs,
            mlir::Location loc) -> std::optional<mlir::Value> {
          if (inputs.size() != 1)
            return std::nullopt;
          return inputs[0];
        });

    addSourceMaterialization(
        [&](mlir::OpBuilder &builder, mlir::Type resultType,
            mlir::ValueRange inputs,
            mlir::Location loc) -> std::optional<mlir::Value> {
          if (inputs.size() != 1)
            return std::nullopt;
          return inputs[0];
        });
  }
};

struct FuncModulePortInfo {

  static const size_t NUM_MEM_INPUTS = 1;
  static const size_t NUM_MEM_OUTPUTS = 5;

  ModulePortInfo ports;
  std::vector<std::pair<size_t, size_t>> memIO;

  FuncModulePortInfo() : ports({}, {}){};

  void addMemIO(Value arg, std::string name, MLIRContext *ctx);

  size_t getNumMemOutputs() { return NUM_MEM_OUTPUTS * memIO.size(); }
};

} // namespace

void FuncModulePortInfo::addMemIO(Value arg, std::string name,
                                  MLIRContext *ctx) {
  // Function argument must be of type memref
  auto memrefType = dyn_cast<MemRefType>(arg.getType());
  assert(memrefType);

  // Types used by memory IO
  Type i1Type = IntegerType::get(ctx, 1);
  Type addrType = IntegerType::get(ctx, 32); // todo: hardcoded to 32 for now
  Type dataType = memrefType.getElementType();

  // Remember ports which correspond to memory
  auto inputIdx = ports.inputs.size();
  auto outputIdx = ports.outputs.size();
  memIO.push_back(std::make_pair(inputIdx, outputIdx));

  // Load data input
  ports.inputs.push_back({StringAttr::get(ctx, name + "_load_data"),
                          hw::PortDirection::INPUT, dataType, inputIdx,
                          hw::InnerSymAttr{}});
  // Load enable output
  ports.outputs.push_back({StringAttr::get(ctx, name + "_load_en"),
                           hw::PortDirection::OUTPUT, i1Type, outputIdx++,
                           hw::InnerSymAttr{}});
  // Load address output
  ports.outputs.push_back({StringAttr::get(ctx, name + "_load_addr"),
                           hw::PortDirection::OUTPUT, addrType, outputIdx++,
                           hw::InnerSymAttr{}});
  // Store enable output
  ports.outputs.push_back({StringAttr::get(ctx, name + "_store_en"),
                           hw::PortDirection::OUTPUT, i1Type, outputIdx++,
                           hw::InnerSymAttr{}});
  // Store address output
  ports.outputs.push_back({StringAttr::get(ctx, name + "_store_addr"),
                           hw::PortDirection::OUTPUT, addrType, outputIdx++,
                           hw::InnerSymAttr{}});
  // Store data output
  ports.outputs.push_back({StringAttr::get(ctx, name + "_store_data"),
                           hw::PortDirection::OUTPUT, dataType, outputIdx,
                           hw::InnerSymAttr{}});
}

/// Returns a submodule name resulting from an operation, without discriminating
/// type information.
static std::string getBareSubModuleName(Operation *oldOp) {
  std::string subModuleName = oldOp->getName().getStringRef().str();
  std::replace(subModuleName.begin(), subModuleName.end(), '.', '_');
  return subModuleName;
}

static std::string getCallName(Operation *op) {
  auto callOp = dyn_cast<handshake::InstanceOp>(op);
  return callOp ? callOp.getModule().str() : getBareSubModuleName(op);
}

/// Extracts the type of the data-carrying type of opType. If opType is an ESI
/// channel, getHandshakeBundleDataType extracts the data-carrying type, else,
/// assume that opType itself is the data-carrying type.
static Type getOperandDataType(Value op) {
  auto opType = op.getType();
  if (auto channelType = opType.dyn_cast<esi::ChannelType>())
    return channelType.getInner();
  return opType;
}

/// Returns a set of types which may uniquely identify the provided op. Return
/// value is <inputTypes, outputTypes>.
using DiscriminatingTypes = std::pair<SmallVector<Type>, SmallVector<Type>>;
static DiscriminatingTypes getHandshakeDiscriminatingTypes(Operation *op) {
  return TypeSwitch<Operation *, DiscriminatingTypes>(op)
      .Case<MemoryOp, ExternalMemoryOp>([&](auto memOp) {
        return DiscriminatingTypes{{},
                                   {memOp.getMemRefType().getElementType()}};
      })
      .Default([&](auto) {
        // By default, all in- and output types which is not a control type
        // (NoneType) are discriminating types.
        SmallVector<Type> inTypes, outTypes;
        llvm::transform(op->getOperands(), std::back_inserter(inTypes),
                        getOperandDataType);
        llvm::transform(op->getResults(), std::back_inserter(outTypes),
                        getOperandDataType);
        return DiscriminatingTypes{inTypes, outTypes};
      });
}

/// Get type name. Currently we only support integer or index types.
// NOLINTNEXTLINE(misc-no-recursion)
static std::string getTypeName(Location loc, Type type) {
  std::string typeName;
  // Builtin types
  if (type.isIntOrIndex()) {
    if (auto indexType = type.dyn_cast<IndexType>())
      typeName += "_ui" + std::to_string(indexType.kInternalStorageBitWidth);
    else if (type.isSignedInteger())
      typeName += "_si" + std::to_string(type.getIntOrFloatBitWidth());
    else
      typeName += "_ui" + std::to_string(type.getIntOrFloatBitWidth());
  } else if (isa<FloatType>(type))
    typeName += "_f" + std::to_string(type.getIntOrFloatBitWidth());
  else if (auto tupleType = type.dyn_cast<TupleType>()) {
    typeName += "_tuple";
    for (auto elementType : tupleType.getTypes())
      typeName += getTypeName(loc, elementType);
  } else if (auto structType = type.dyn_cast<hw::StructType>()) {
    typeName += "_struct";
    for (auto element : structType.getElements())
      typeName += "_" + element.name.str() + getTypeName(loc, element.type);
  } else if (isa<NoneType>(type))
    typeName += "_none";
  else
    emitError(loc) << "unsupported data type '" << type << "'";

  return typeName;
}

/// Construct a name for creating HW sub-module.
static std::string getSubModuleName(Operation *oldOp) {
  if (auto instanceOp = dyn_cast<handshake::InstanceOp>(oldOp); instanceOp)
    return instanceOp.getModule().str();

  std::string subModuleName = getBareSubModuleName(oldOp);

  // Add value of the constant operation.
  if (auto constOp = dyn_cast<handshake::ConstantOp>(oldOp)) {
    if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>()) {
      auto intType = intAttr.getType();

      if (intType.isSignedInteger())
        subModuleName += "_c" + std::to_string(intAttr.getSInt());
      else if (intType.isUnsignedInteger())
        subModuleName += "_c" + std::to_string(intAttr.getUInt());
      else
        subModuleName += "_c" + std::to_string((uint64_t)intAttr.getInt());
    } else if (auto floatAttr = constOp.getValue().dyn_cast<FloatAttr>())
      subModuleName +=
          "_c" + std::to_string(floatAttr.getValue().convertToFloat());
    else
      oldOp->emitError("unsupported constant type");
  }

  // Add discriminating in- and output types.
  auto [inTypes, outTypes] = getHandshakeDiscriminatingTypes(oldOp);
  if (!inTypes.empty())
    subModuleName += "_in";
  for (auto inType : inTypes)
    subModuleName += getTypeName(oldOp->getLoc(), inType);

  if (!outTypes.empty())
    subModuleName += "_out";
  for (auto outType : outTypes)
    subModuleName += getTypeName(oldOp->getLoc(), outType);

  // Add memory ID.
  if (auto memOp = dyn_cast<handshake::MemoryOp>(oldOp))
    subModuleName += "_id" + std::to_string(memOp.getId());

  // Add compare kind.
  if (auto comOp = dyn_cast<mlir::arith::CmpIOp>(oldOp))
    subModuleName += "_" + stringifyEnum(comOp.getPredicate()).str();

  // Add buffer information.
  if (auto bufferOp = dyn_cast<handshake::BufferOp>(oldOp)) {
    subModuleName += "_" + std::to_string(bufferOp.getNumSlots()) + "slots";
    if (bufferOp.isSequential())
      subModuleName += "_seq";
    else
      subModuleName += "_fifo";

    if (auto initValues = bufferOp.getInitValues()) {
      subModuleName += "_init";
      for (const Attribute e : *initValues) {
        assert(e.isa<IntegerAttr>());
        subModuleName +=
            "_" + std::to_string(e.dyn_cast<IntegerAttr>().getInt());
      }
    }
  }

  // Add control information.
  if (auto ctrlInterface = dyn_cast<handshake::ControlInterface>(oldOp);
      ctrlInterface && ctrlInterface.isControl()) {
    // Add some additional discriminating info for non-typed operations.
    subModuleName += "_" + std::to_string(oldOp->getNumOperands()) + "ins_" +
                     std::to_string(oldOp->getNumResults()) + "outs";
    subModuleName += "_ctrl";
  }

  return subModuleName;
}

//===----------------------------------------------------------------------===//
// HW Sub-module Related Functions
//===----------------------------------------------------------------------===//

/// Check whether a submodule with the same name has been created elsewhere in
/// the top level module. Return the matched module operation if true, otherwise
/// return nullptr.
static Operation *checkSubModuleOp(mlir::ModuleOp parentModule,
                                   StringRef modName) {
  if (auto mod = parentModule.lookupSymbol<HWModuleOp>(modName))
    return mod;
  if (auto mod = parentModule.lookupSymbol<HWModuleExternOp>(modName))
    return mod;
  return nullptr;
}

static Operation *checkSubModuleOp(mlir::ModuleOp parentModule,
                                   Operation *oldOp) {
  auto *moduleOp = checkSubModuleOp(parentModule, getSubModuleName(oldOp));

  if (isa<handshake::InstanceOp>(oldOp))
    assert(moduleOp &&
           "handshake.instance target modules should always have been lowered "
           "before the modules that reference them!");
  return moduleOp;
}

//===----------------------------------------------------------------------===//
// Port-Generating Functions
//===----------------------------------------------------------------------===//

static ModulePortInfo getPortInfo(Operation *op) {
  return getPortInfoForOpTypes(op, op->getOperandTypes(), op->getResultTypes());
}

static FuncModulePortInfo getFuncPortInfo(handshake::FuncOp funcOp) {

  FuncModulePortInfo info;
  auto *ctx = funcOp.getContext();
  Type i1Type = IntegerType::get(ctx, 1);

  // Add all outputs of function
  TypeRange outputs = funcOp.getResultTypes();
  for (auto &[idx, res] : llvm::enumerate(outputs))
    info.ports.outputs.push_back({funcOp.getResName(idx),
                                  hw::PortDirection::OUTPUT, esiWrapper(res),
                                  idx, hw::InnerSymAttr{}});

  // Add all inputs of function, replacing memrefs with appropriate ports
  for (auto &[idx, arg] : llvm::enumerate(funcOp.getArguments()))
    if (isa<MemRefType>(arg.getType()))
      info.addMemIO(arg, funcOp.getArgName(idx).str(), ctx);
    else
      info.ports.inputs.push_back(
          {funcOp.getArgName(idx), hw::PortDirection::INPUT,
           esiWrapper(arg.getType()), idx, hw::InnerSymAttr{}});

  // Add clock and reset signals to inputs
  unsigned numInputs = info.ports.inputs.size();
  info.ports.inputs.push_back({StringAttr::get(ctx, "clock"),
                               hw::PortDirection::INPUT, i1Type, numInputs,
                               hw::InnerSymAttr{}});
  info.ports.inputs.push_back({StringAttr::get(ctx, "reset"),
                               hw::PortDirection::INPUT, i1Type, numInputs + 1,
                               hw::InnerSymAttr{}});

  return info;
}

static ModulePortInfo getMemPortInfo(handshake::MemoryControllerOp memOp,
                                     FuncModulePortInfo &info, size_t memIdx) {

  ModulePortInfo ports({}, {});
  auto *ctx = memOp.getContext();
  Type i1Type = IntegerType::get(ctx, 1);

  auto &[inFuncPortIdx, outFuncPortIdx] = info.memIO[memIdx];

  // Add input ports coming from outside the containing module
  size_t inPortIdx = 0;
  for (size_t i = inFuncPortIdx,
              e = inFuncPortIdx + FuncModulePortInfo::NUM_MEM_INPUTS;
       i < e; i++) {
    auto funcInput = info.ports.inputs[i];
    ports.inputs.push_back({funcInput.name, hw::PortDirection::INPUT,
                            funcInput.type, inPortIdx++, hw::InnerSymAttr{}});
  }

  // Add input ports corresponding to memory interface operands
  for (auto &[idx, arg] : llvm::enumerate(memOp.getInputs()))
    ports.inputs.push_back({StringAttr::get(ctx, memOp.getOperandName(idx + 1)),
                            hw::PortDirection::INPUT, esiWrapper(arg.getType()),
                            inPortIdx++, hw::InnerSymAttr{}});

  // Add clock and reset signals to inputs
  ports.inputs.push_back({StringAttr::get(ctx, "clock"),
                          hw::PortDirection::INPUT, i1Type, inPortIdx++,
                          hw::InnerSymAttr{}});
  ports.inputs.push_back({StringAttr::get(ctx, "reset"),
                          hw::PortDirection::INPUT, i1Type, inPortIdx,
                          hw::InnerSymAttr{}});

  // Add output ports corresponding to memory interface operands
  size_t outPortIdx = 0;
  for (auto &[idx, arg] : llvm::enumerate(memOp.getResults()))
    ports.outputs.push_back({StringAttr::get(ctx, memOp.getResultName(idx)),
                             hw::PortDirection::OUTPUT,
                             esiWrapper(arg.getType()), outPortIdx++,
                             hw::InnerSymAttr{}});

  // Add output ports going outside the containing module
  for (size_t i = outFuncPortIdx,
              e = outFuncPortIdx + FuncModulePortInfo::NUM_MEM_OUTPUTS;
       i < e; i++) {
    auto funcOutput = info.ports.outputs[i];
    ports.outputs.push_back({funcOutput.name, hw::PortDirection::OUTPUT,
                             funcOutput.type, outPortIdx++,
                             hw::InnerSymAttr{}});
  }

  return ports;
}

static ModulePortInfo getEndPortInfo(handshake::EndOp endOp,
                                     FuncModulePortInfo &info) {

  ModulePortInfo ports({}, {});
  auto *ctx = endOp.getContext();
  Type i1Type = IntegerType::get(ctx, 1);

  // Add input ports corresponding to end operands
  size_t inPortIdx = 0;
  for (auto &[idx, arg] : llvm::enumerate(endOp.getOperands()))
    ports.inputs.push_back({StringAttr::get(ctx, "in" + std::to_string(idx)),
                            hw::PortDirection::INPUT, esiWrapper(arg.getType()),
                            inPortIdx++, hw::InnerSymAttr{}});

  // Add clock and reset signals to inputs
  ports.inputs.push_back({StringAttr::get(ctx, "clock"),
                          hw::PortDirection::INPUT, i1Type, inPortIdx++,
                          hw::InnerSymAttr{}});
  ports.inputs.push_back({StringAttr::get(ctx, "reset"),
                          hw::PortDirection::INPUT, i1Type, inPortIdx,
                          hw::InnerSymAttr{}});

  // Add output ports corresponding to function return values
  auto numReturnValues = info.ports.outputs.size() - info.getNumMemOutputs();
  auto returnValOperands = endOp.getOperands().take_front(numReturnValues);
  for (auto &[idx, arg] : llvm::enumerate(returnValOperands))
    ports.outputs.push_back({StringAttr::get(ctx, "out" + std::to_string(idx)),
                             hw::PortDirection::OUTPUT,
                             esiWrapper(arg.getType()), idx,
                             hw::InnerSymAttr{}});

  return ports;
}

namespace {

class FuncOpConversionPattern : public OpConversionPattern<handshake::FuncOp> {
public:
  FuncOpConversionPattern(MLIRContext *ctx, HandshakeLoweringState &ls)
      : OpConversionPattern<handshake::FuncOp>(ctx, 1), ls(ls) {}

  LogicalResult
  matchAndRewrite(handshake::FuncOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    FuncModulePortInfo info = getFuncPortInfo(op);

    if (op.isExternal())
      ls.extModBuilder.create<hw::HWModuleExternOp>(
          op.getLoc(), ls.extModBuilder.getStringAttr(op.getName()),
          info.ports);
    else {
      // Create module for the function
      auto mod = ls.extModBuilder.create<hw::HWModuleOp>(
          op.getLoc(), ls.extModBuilder.getStringAttr(op.getName()),
          info.ports);

      // Replace uses of function arguments with module inputs
      for (auto it : llvm::zip(op.getArguments(), mod.getArguments()))
        std::get<0>(it).replaceAllUsesWith(std::get<1>(it));

      // Move all operations to the new module
      auto &moduleBlockOps = mod.getBodyBlock()->getOperations();
      moduleBlockOps.splice(moduleBlockOps.begin(),
                            op.getBody().front().getOperations());

      // Insert a start buffer for each channel-typed input
      bufferInputs(mod, rewriter);

      // Convert memory interfaces (and connect them with function IO)
      auto memInstances = convertMemories(mod, info, rewriter);

      // Convert end operation (add results to represent function return values)
      auto endInstance = convertEnd(mod, info, rewriter);

      // Set operands of output operation to match module outputs
      setModuleOutputs(mod, memInstances, endInstance);
    }

    // Original function can safely be deleted before returning
    rewriter.eraseOp(op);
    return success();
  }

private:
  HandshakeLoweringState &ls;

  void bufferInputs(HWModuleOp mod, ConversionPatternRewriter &rewriter) const {

    unsigned argIdx = 0;
    rewriter.setInsertionPointToStart(mod.getBodyBlock());

    for (auto arg : mod.getArguments()) {
      auto argType = arg.getType();
      if (!isa<esi::ChannelType>(argType))
        continue;

      // Create ports for input buffer
      ModulePortInfo ports({}, {});
      auto *ctx = mod.getContext();
      Type i1Type = IntegerType::get(ctx, 1);

      // Function argument input
      ports.inputs.push_back({StringAttr::get(ctx, "in0"),
                              hw::PortDirection::INPUT, argType, 0,
                              hw::InnerSymAttr{}});
      // Clock and reset inputs
      ports.inputs.push_back({StringAttr::get(ctx, "clock"),
                              hw::PortDirection::INPUT, i1Type, 1,
                              hw::InnerSymAttr{}});
      ports.inputs.push_back({StringAttr::get(ctx, "reset"),
                              hw::PortDirection::INPUT, i1Type, 2,
                              hw::InnerSymAttr{}});
      // Module output matches function argument type
      ports.outputs.push_back({StringAttr::get(ctx, "out0"),
                               hw::PortDirection::OUTPUT, argType, 0,
                               hw::InnerSymAttr{}});

      // Check whether we need to create a new external module
      auto argLoc = arg.getLoc();
      std::string modName = "handshake_start";
      if (auto channelType = argType.dyn_cast<esi::ChannelType>())
        modName += getTypeName(argLoc, channelType.getInner());
      hw::HWModuleLike extModule = checkSubModuleOp(ls.parentModule, modName);
      if (!extModule)
        extModule = ls.extModBuilder.create<hw::HWModuleExternOp>(
            argLoc, ls.extModBuilder.getStringAttr(modName), ports);

      // Replace operation with corresponding hardware module instance
      llvm::SmallVector<Value> operands;
      operands.push_back(arg);
      operands.push_back(mod.getArgument(mod.getNumArguments() - 2));
      operands.push_back(mod.getArgument(mod.getNumArguments() - 1));
      auto instanceOp = rewriter.create<hw::InstanceOp>(
          argLoc, extModule,
          rewriter.getStringAttr("handshake_start" + std::to_string(argIdx++)),
          operands);

      arg.replaceAllUsesWith(instanceOp.getResult(0));
    }
  }

  SmallVector<hw::InstanceOp>
  convertMemories(HWModuleOp mod, FuncModulePortInfo &info,
                  ConversionPatternRewriter &rewriter) const {

    SmallVector<hw::InstanceOp> instances;

    for (auto &[memIdx, portIndices] : llvm::enumerate(info.memIO)) {
      // Identify the memory controller refering to the memref
      auto user = *mod.getArgument(portIndices.first).getUsers().begin();
      assert(user && "old memref value should have a user");
      auto memOp = dyn_cast<handshake::MemoryControllerOp>(user);
      assert(memOp && "user of old memref value should be memory interface");

      std::string memName = getSubModuleName(memOp);
      auto ports = getMemPortInfo(memOp, info, memIdx);

      // Create an external module definition if necessary
      hw::HWModuleLike extModule = checkSubModuleOp(ls.parentModule, memName);
      if (!extModule)
        extModule = ls.extModBuilder.create<hw::HWModuleExternOp>(
            memOp->getLoc(), ls.extModBuilder.getStringAttr(memName), ports);

      // Combine memory inputs from the function and internal memory inputs into
      // the new instance operands
      SmallVector<Value> instOperands;
      for (auto i = portIndices.first,
                e = portIndices.first + FuncModulePortInfo::NUM_MEM_INPUTS;
           i < e; i++)
        instOperands.push_back(mod.getArgument(i));
      instOperands.insert(instOperands.end(), memOp.getInputs().begin(),
                          memOp.getInputs().end());
      instOperands.push_back(mod.getArgument(mod.getNumArguments() - 2));
      instOperands.push_back(mod.getArgument(mod.getNumArguments() - 1));

      // Create instance of memory interface
      rewriter.setInsertionPoint(memOp);
      auto instance = rewriter.create<hw::InstanceOp>(
          memOp.getLoc(), extModule,
          rewriter.getStringAttr(ls.nameUniquer(memOp)), instOperands);

      // Replace uses of memory interface results with new instance results
      for (auto it :
           llvm::zip(memOp->getResults(),
                     instance->getResults().take_front(memOp->getNumResults())))
        std::get<0>(it).replaceAllUsesWith(std::get<1>(it));

      rewriter.eraseOp(memOp);
      instances.push_back(instance);
    }

    return instances;
  }

  hw::InstanceOp convertEnd(HWModuleOp mod, FuncModulePortInfo &info,
                            ConversionPatternRewriter &rewriter) const {
    auto endOp = *mod.getBodyBlock()->getOps<handshake::EndOp>().begin();

    auto ports = getEndPortInfo(endOp, info);

    // Create external module
    auto extModule = ls.extModBuilder.create<hw::HWModuleExternOp>(
        endOp->getLoc(),
        ls.extModBuilder.getStringAttr(getSubModuleName(endOp)), ports);

    // Create instance of end operation
    SmallVector<Value> instOperands(endOp.getOperands());
    instOperands.push_back(mod.getArgument(mod.getNumArguments() - 2));
    instOperands.push_back(mod.getArgument(mod.getNumArguments() - 1));
    rewriter.setInsertionPoint(endOp);
    auto instance = rewriter.create<hw::InstanceOp>(
        endOp.getLoc(), extModule,
        rewriter.getStringAttr(ls.nameUniquer(endOp)), instOperands);

    rewriter.eraseOp(endOp);
    return instance;
  }

  void setModuleOutputs(HWModuleOp mod,
                        SmallVector<hw::InstanceOp> memInstances,
                        hw::InstanceOp endInstance) const {
    auto outputOp = *mod.getBodyBlock()->getOps<hw::OutputOp>().begin();
    SmallVector<Value> newOperands;
    newOperands.insert(newOperands.end(), endInstance.getResults().begin(),
                       endInstance.getResults().end());
    for (auto &mem : memInstances) {
      auto memOutputs =
          mem.getResults().take_back(FuncModulePortInfo::NUM_MEM_OUTPUTS);
      newOperands.insert(newOperands.end(), memOutputs.begin(),
                         memOutputs.end());
    }
    outputOp->setOperands(newOperands);
  }
};

template <typename T>
class ExtModuleConversionPattern : public OpConversionPattern<T> {
public:
  ExtModuleConversionPattern(ESITypeConverter &typeConverter, MLIRContext *ctx,
                             HandshakeLoweringState &ls)
      : OpConversionPattern<T>::OpConversionPattern(typeConverter, ctx),
        ls(ls) {}
  using OpAdaptor = typename T::Adaptor;

  LogicalResult
  matchAndRewrite(T op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Check whether we need to create a new external module
    hw::HWModuleLike extModule = checkSubModuleOp(ls.parentModule, op);
    if (!extModule)
      extModule = ls.extModBuilder.create<hw::HWModuleExternOp>(
          op.getLoc(), ls.extModBuilder.getStringAttr(getSubModuleName(op)),
          getPortInfo(op));

    // Replace operation with corresponding hardware module instance
    llvm::SmallVector<Value> operands = adaptor.getOperands();
    if (op.template hasTrait<mlir::OpTrait::HasClock>()) {
      auto parent = cast<hw::HWModuleOp>(op->getParentOp());
      operands.push_back(parent.getArgument(parent.getNumArguments() - 2));
      operands.push_back(parent.getArgument(parent.getNumArguments() - 1));
    }
    auto instanceOp = rewriter.replaceOpWithNewOp<hw::InstanceOp>(
        op, extModule, rewriter.getStringAttr(ls.nameUniquer(op)), operands);

    // Replace operation results with new instance results
    for (auto it : llvm::zip(op->getResults(), instanceOp->getResults()))
      std::get<0>(it).replaceAllUsesWith(std::get<1>(it));

    return success();
  }

protected:
  HandshakeLoweringState &ls;
};
} // namespace

namespace {
class HandshakeToNetListPass
    : public HandshakeToNetlistBase<HandshakeToNetListPass> {
public:
  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();

    // We only support one function per module
    auto functions = mod.getOps<handshake::FuncOp>();
    if (++functions.begin() != functions.end()) {
      mod->emitOpError()
          << "we currently only support one handshake function per module";
      return signalPassFailure();
    }
    handshake::FuncOp funcOp = *functions.begin();

    // Lowering to HW requires that every value is used exactly once.
    // Check whether this precondition is met, and if not, exit
    if (failed(verifyAllValuesHasOneUse(funcOp))) {
      funcOp.emitOpError() << "not all values are used exactly once";
      return signalPassFailure();
    }

    // Create helper struct for lowering
    std::map<std::string, unsigned> instanceNameCntr;
    NameUniquer instanceUniquer = [&](Operation *op) {
      std::string instName = getCallName(op);
      return instName + std::to_string(instanceNameCntr[instName]++);
    };
    auto ls = HandshakeLoweringState{mod, instanceUniquer};

    ESITypeConverter typeConverter;
    MLIRContext &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    ConversionTarget target(ctx);
    patterns.insert<FuncOpConversionPattern>(&ctx, ls);
    patterns.insert<
        // Handshake operations
        ExtModuleConversionPattern<handshake::ConditionalBranchOp>,
        ExtModuleConversionPattern<handshake::BranchOp>,
        ExtModuleConversionPattern<handshake::MergeOp>,
        ExtModuleConversionPattern<handshake::ControlMergeOp>,
        ExtModuleConversionPattern<handshake::MuxOp>,
        ExtModuleConversionPattern<handshake::SelectOp>,
        ExtModuleConversionPattern<handshake::SourceOp>,
        ExtModuleConversionPattern<handshake::ConstantOp>,
        ExtModuleConversionPattern<handshake::SinkOp>,
        ExtModuleConversionPattern<handshake::ForkOp>,
        ExtModuleConversionPattern<handshake::DynamaticReturnOp>,
        ExtModuleConversionPattern<handshake::DynamaticLoadOp>,
        ExtModuleConversionPattern<handshake::DynamaticStoreOp>,
        // Arith operations
        ExtModuleConversionPattern<arith::AddFOp>,
        ExtModuleConversionPattern<arith::AddIOp>,
        ExtModuleConversionPattern<arith::AndIOp>,
        ExtModuleConversionPattern<arith::BitcastOp>,
        ExtModuleConversionPattern<arith::CeilDivSIOp>,
        ExtModuleConversionPattern<arith::CeilDivUIOp>,
        ExtModuleConversionPattern<arith::CmpFOp>,
        ExtModuleConversionPattern<arith::CmpIOp>,
        ExtModuleConversionPattern<arith::DivFOp>,
        ExtModuleConversionPattern<arith::DivSIOp>,
        ExtModuleConversionPattern<arith::DivUIOp>,
        ExtModuleConversionPattern<arith::ExtFOp>,
        ExtModuleConversionPattern<arith::ExtSIOp>,
        ExtModuleConversionPattern<arith::ExtUIOp>,
        ExtModuleConversionPattern<arith::FPToSIOp>,
        ExtModuleConversionPattern<arith::FPToUIOp>,
        ExtModuleConversionPattern<arith::FloorDivSIOp>,
        ExtModuleConversionPattern<arith::IndexCastOp>,
        ExtModuleConversionPattern<arith::IndexCastUIOp>,
        ExtModuleConversionPattern<arith::MulFOp>,
        ExtModuleConversionPattern<arith::MulIOp>,
        ExtModuleConversionPattern<arith::NegFOp>,
        ExtModuleConversionPattern<arith::OrIOp>,
        ExtModuleConversionPattern<arith::RemFOp>,
        ExtModuleConversionPattern<arith::RemSIOp>,
        ExtModuleConversionPattern<arith::RemUIOp>,
        ExtModuleConversionPattern<arith::SIToFPOp>,
        ExtModuleConversionPattern<arith::ShLIOp>,
        ExtModuleConversionPattern<arith::ShRSIOp>,
        ExtModuleConversionPattern<arith::ShRUIOp>,
        ExtModuleConversionPattern<arith::SubFOp>,
        ExtModuleConversionPattern<arith::SubIOp>,
        ExtModuleConversionPattern<arith::TruncFOp>,
        ExtModuleConversionPattern<arith::TruncIOp>,
        ExtModuleConversionPattern<arith::UIToFPOp>,
        ExtModuleConversionPattern<arith::XOrIOp>>(typeConverter,
                                                   funcOp->getContext(), ls);

    target.addLegalOp<hw::HWModuleOp, hw::HWModuleExternOp, hw::OutputOp,
                      hw::InstanceOp>();
    target.addIllegalDialect<handshake::HandshakeDialect, arith::ArithDialect,
                             memref::MemRefDialect>();

    if (failed(applyPartialConversion(funcOp, target, std::move(patterns))))
      return signalPassFailure();
  }
};
} // end anonymous namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createHandshakeToNetlistPass() {
  return std::make_unique<HandshakeToNetListPass>();
}
