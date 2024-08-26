//===- HandshakeOps.cpp - Handshake operations ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file originates from the CIRCT project (https://github.com/llvm/circt).
// It includes modifications made as part of Dynamatic.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the Handshake operations struct.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Utils/Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

static ParseResult parseHandshakeType(OpAsmParser &parser, Type &type) {
  return parser.parseCustomTypeWithFallback(type, [&](Type &ty) -> ParseResult {
    if ((ty = handshake::detail::jointHandshakeTypeParser(parser)))
      return success();
    return failure();
  });
}

static ParseResult parseHandshakeTypes(OpAsmParser &parser,
                                       SmallVectorImpl<Type> &types) {
  do {
    if (parseHandshakeType(parser, types.emplace_back()))
      return failure();
  } while (!parser.parseOptionalComma());
  return success();
}

static void printHandshakeType(OpAsmPrinter &printer, Operation * /*op*/,
                               Type type) {
  if (auto controlType = dyn_cast<handshake::ControlType>(type)) {
    controlType.print(printer);
  } else if (auto channelType = dyn_cast<handshake::ChannelType>(type)) {
    channelType.print(printer);
  } else {
    llvm_unreachable("not a handshake type");
  }
}

static void printHandshakeTypes(OpAsmPrinter &printer, Operation * /*op*/,
                                TypeRange types) {
  if (types.empty())
    return;
  for (Type ty : types.drop_back()) {
    printHandshakeType(printer, nullptr, ty);
    printer << ", ";
  }
  printHandshakeType(printer, nullptr, types.back());
}

static void printHandshakeType(OpAsmPrinter &printer, Type type) {
  printHandshakeType(printer, nullptr, type);
}

static ParseResult parseSingleTypedHandshakeOp(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &operand,
    NamedAttrList &attributes, Type &type, SmallVectorImpl<Type> &resTypes) {
  // Parse the operation's size between square brackets
  unsigned size;
  if (parser.parseLSquare() || parser.parseInteger(size) ||
      parser.parseRSquare())
    return failure();

  // Parse the single operand and attribute dictionnary
  if (parser.parseOperand(operand) || parser.parseOptionalAttrDict(attributes))
    return failure();

  // Parse the single handshake type common to all operands and results
  if (parser.parseColon() || parseHandshakeType(parser, type))
    return failure();
  resTypes.assign(size, type);
  return success();
}

static void printSingleTypedHandshakeOp(OpAsmPrinter &printer, Operation *op,
                                        Value operand,
                                        DictionaryAttr attributes, Type type,
                                        TypeRange resTypes) {
  printer << "[" << resTypes.size() << "] " << operand;
  printer.printOptionalAttrDict(attributes.getValue());
  printer << " : ";
  printHandshakeType(printer, type);
}

/// Verifies whether an indexing value is wide enough to index into a provided
/// number of operands.
static LogicalResult verifyIndexWideEnough(Operation *op, Value indexVal,
                                           uint64_t numOperands) {
  auto idxType = dyn_cast<handshake::ChannelType>(indexVal.getType());
  if (!idxType) {
    return op->emitError() << "expected index value to be of type "
                              "handshake::ChannelType, but got "
                           << indexVal.getType();
  }
  unsigned idxWidth = idxType.getDataBitWidth();

  // Check whether the bitwidth can support the provided number of operands
  if (idxWidth < 64) {
    uint64_t maxNumOperands = (uint64_t)1 << idxWidth;
    if (numOperands > maxNumOperands) {
      return op->emitError()
             << "bitwidth of indexing value is " << idxWidth
             << ", which can index into " << maxNumOperands
             << " operands, but found " << numOperands << " operands";
    }
  }
  return success();
}

static bool isControlCheckTypeAndOperand(Type dataType, Value operand) {
  // The operation is a control operation if its operand data type is a
  // control-only channel
  if (isa<handshake::ControlType>(dataType))
    return true;

  // Otherwise, the operation is a control operation if the operation's
  // operand originates from the control network
  auto *defOp = operand.getDefiningOp();
  return isa_and_nonnull<ControlMergeOp>(defOp) &&
         operand == defOp->getResult(0);
}

IntegerType
dynamatic::handshake::getOptimizedIndexValType(OpBuilder &builder,
                                               unsigned numToIndex) {
  return builder.getIntegerType(std::max(
      1U, APInt(APInt::APINT_BITS_PER_WORD, numToIndex).ceilLogBase2()));
}

//===----------------------------------------------------------------------===//
// TableGen'd canonicalization patterns
//===----------------------------------------------------------------------===//

static unsigned getDataBitWidth(Value val) {
  return cast<handshake::ChannelType>(val.getType()).getDataBitWidth();
}

namespace {
#include "lib/Dialect/Handshake/HandshakeCanonicalization.inc"
} // namespace

//===----------------------------------------------------------------------===//
// BufferOp
//===----------------------------------------------------------------------===//

void BufferOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                     Value operand, const TimingInfo &timing,
                     std::optional<unsigned> numSlots) {
  odsState.addOperands(operand);
  odsState.addTypes(operand.getType());

  // Create attribute dictionary
  SmallVector<NamedAttribute> attributes;
  MLIRContext *ctx = odsState.getContext();
  attributes.emplace_back(StringAttr::get(ctx, TIMING_ATTR_NAME),
                          TimingAttr::get(ctx, timing));
  if (numSlots) {
    attributes.emplace_back(
        StringAttr::get(ctx, NUM_SLOTS_ATTR_NAME),
        IntegerAttr::get(IntegerType::get(ctx, 32, IntegerType::Unsigned),
                         *numSlots));
  }

  odsState.addAttribute(RTL_PARAMETERS_ATTR_NAME,
                        DictionaryAttr::get(ctx, attributes));
}

std::pair<handshake::ChannelType, bool> BufferOp::getReshapableChannelType() {
  return {dyn_cast<handshake::ChannelType>(getOperand().getType()), true};
}
//===----------------------------------------------------------------------===//
// MergeOp
//===----------------------------------------------------------------------===//

OpResult MergeOp::getDataResult() { return cast<OpResult>(getResult()); }

//===----------------------------------------------------------------------===//
// MuxOp
//===----------------------------------------------------------------------===//

LogicalResult
MuxOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                        ValueRange operands, DictionaryAttr attributes,
                        mlir::OpaqueProperties properties,
                        mlir::RegionRange regions,
                        SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  // MuxOp must have at least one data operand (in addition to the select
  // operand)
  if (operands.size() < 2)
    return failure();
  // Result type is type of any data operand
  inferredReturnTypes.push_back(operands[1].getType());
  return success();
}

bool MuxOp::isControl() {
  return isa<handshake::ControlType>(getResult().getType());
}

ParseResult MuxOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand selectOperand;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> allOperands;
  handshake::ChannelType selectType;
  Type dataType;
  SmallVector<Type, 2> dataOperandsTypes;
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperand(selectOperand) || parser.parseLSquare() ||
      parser.parseOperandList(allOperands) || parser.parseRSquare() ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseCustomTypeWithFallback(selectType) || parser.parseComma() ||
      parseHandshakeType(parser, dataType))
    return failure();

  int size = allOperands.size();
  dataOperandsTypes.assign(size, dataType);
  result.addTypes(dataType);
  allOperands.insert(allOperands.begin(), selectOperand);
  if (parser.resolveOperands(
          allOperands,
          llvm::concat<const Type>(ArrayRef<Type>(selectType),
                                   ArrayRef<Type>(dataOperandsTypes)),
          allOperandLoc, result.operands))
    return failure();
  return success();
}

void MuxOp::print(OpAsmPrinter &p) {
  OperandRange operands = getOperands();
  p << ' ' << operands.front();
  p << " [";
  p.printOperands(operands.drop_front());
  p << "]";
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : ";
  p.printStrippedAttrOrType(getSelectOperand().getType());
  p << ", ";
  printHandshakeType(p, getResult().getType());
}

LogicalResult MuxOp::verify() {
  return verifyIndexWideEnough(*this, getSelectOperand(),
                               getDataOperands().size());
}

OpResult MuxOp::getDataResult() { return cast<OpResult>(getResult()); }

//===----------------------------------------------------------------------===//
// ControlMergeOp
//===----------------------------------------------------------------------===//

ParseResult ControlMergeOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  Type dataType;
  handshake::ChannelType indexType;
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(operands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parseHandshakeType(parser, dataType) || parser.parseComma() ||
      parser.parseCustomTypeWithFallback(indexType))
    return failure();

  SmallVector<Type> operandTypes(operands.size(), dataType);
  result.addTypes({dataType, indexType});
  return parser.resolveOperands(operands, operandTypes, allOperandLoc,
                                result.operands);
}

void ControlMergeOp::print(OpAsmPrinter &p) {
  p << " " << getOperands() << " ";
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : ";
  printHandshakeType(p, getResult().getType());
  p << ", ";
  p.printStrippedAttrOrType(getIndex().getType());
}

LogicalResult ControlMergeOp::verify() {
  TypeRange operandTypes = getOperandTypes();
  if (operandTypes.empty())
    return emitOpError("operation must have at least one operand");
  Type refType = operandTypes.front();
  for (Type type : operandTypes.drop_front()) {
    if (refType != type)
      return emitOpError("all operands should have the same type");
  }
  if (refType != getResult().getType())
    return emitOpError("type of data result should match type of operands");
  return verifyIndexWideEnough(*this, getIndex(), getNumOperands());
}

OpResult ControlMergeOp::getDataResult() { return cast<OpResult>(getResult()); }

LogicalResult FuncOp::verify() {
  // If this function is external there is nothing to do.
  if (isExternal())
    return success();

  // Verify that the argument list of the function and the arg list of the
  // entry block line up.  The trait already verified that the number of
  // arguments is the same between the signature and the block.
  auto fnInputTypes = getArgumentTypes();
  Block &entryBlock = front();

  for (unsigned i = 0, e = entryBlock.getNumArguments(); i != e; ++i)
    if (fnInputTypes[i] != entryBlock.getArgument(i).getType())
      return emitOpError("type of entry block argument #")
             << i << '(' << entryBlock.getArgument(i).getType()
             << ") must match the type of the corresponding argument in "
             << "function signature(" << fnInputTypes[i] << ')';

  // Verify that we have a name for each argument and result of this function.
  auto verifyPortNameAttr = [&](StringRef attrName,
                                unsigned numIOs) -> LogicalResult {
    auto portNamesAttr = (*this)->getAttrOfType<ArrayAttr>(attrName);

    if (!portNamesAttr)
      return emitOpError() << "expected attribute '" << attrName << "'.";

    auto portNames = portNamesAttr.getValue();
    if (portNames.size() != numIOs)
      return emitOpError() << "attribute '" << attrName << "' has "
                           << portNames.size()
                           << " entries but is expected to have " << numIOs
                           << ".";

    if (llvm::any_of(portNames,
                     [&](Attribute attr) { return !attr.isa<StringAttr>(); }))
      return emitOpError() << "expected all entries in attribute '" << attrName
                           << "' to be strings.";

    return success();
  };
  if (failed(verifyPortNameAttr("argNames", getNumArguments())))
    return failure();
  if (failed(verifyPortNameAttr("resNames", getNumResults())))
    return failure();

  return success();
}

/// Parses a FuncOp signature using
/// mlir::function_interface_impl::parseFunctionSignature while getting access
/// to the parsed SSA names to store as attributes.
static ParseResult
parseFuncOpArgs(OpAsmParser &parser,
                SmallVectorImpl<OpAsmParser::Argument> &entryArgs,
                SmallVectorImpl<Type> &resTypes,
                SmallVectorImpl<DictionaryAttr> &resAttrs) {
  bool isVariadic;
  if (mlir::function_interface_impl::parseFunctionSignature(
          parser, /*allowVariadic=*/true, entryArgs, isVariadic, resTypes,
          resAttrs)
          .failed())
    return failure();

  return success();
}

/// Generates names for a handshake.func input and output arguments, based on
/// the number of args as well as a prefix.
static SmallVector<Attribute> getFuncOpNames(Builder &builder, unsigned cnt,
                                             StringRef prefix) {
  SmallVector<Attribute> resNames;
  for (unsigned i = 0; i < cnt; ++i)
    resNames.push_back(builder.getStringAttr(prefix + std::to_string(i)));
  return resNames;
}

void handshake::FuncOp::build(OpBuilder &builder, OperationState &state,
                              StringRef name, FunctionType type,
                              ArrayRef<NamedAttribute> attrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(FuncOp::getFunctionTypeAttrName(state.name),
                     TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());

  if (const auto *argNamesAttrIt = llvm::find_if(
          attrs, [&](auto attr) { return attr.getName() == "argNames"; });
      argNamesAttrIt == attrs.end())
    state.addAttribute("argNames", builder.getArrayAttr({}));

  if (llvm::find_if(attrs, [&](auto attr) {
        return attr.getName() == "resNames";
      }) == attrs.end())
    state.addAttribute("resNames", builder.getArrayAttr({}));

  state.addRegion();
}

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  StringAttr nameAttr;
  SmallVector<OpAsmParser::Argument> args;
  SmallVector<Type> resTypes;
  SmallVector<DictionaryAttr> resAttributes;
  SmallVector<Attribute> argNames;

  // Parse visibility.
  (void)mlir::impl::parseOptionalVisibilityKeyword(parser, result.attributes);

  // Parse signature
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      parseFuncOpArgs(parser, args, resTypes, resAttributes))
    return failure();
  mlir::function_interface_impl::addArgAndResultAttrs(
      builder, result, args, resAttributes,
      handshake::FuncOp::getArgAttrsAttrName(result.name),
      handshake::FuncOp::getResAttrsAttrName(result.name));

  // Set function type
  SmallVector<Type> argTypes;
  for (auto arg : args)
    argTypes.push_back(arg.type);

  result.addAttribute(
      handshake::FuncOp::getFunctionTypeAttrName(result.name),
      TypeAttr::get(builder.getFunctionType(argTypes, resTypes)));

  // Determine the names of the arguments. If no SSA values are present, use
  // fallback names.
  bool noSSANames =
      llvm::any_of(args, [](auto arg) { return arg.ssaName.name.empty(); });
  if (noSSANames) {
    argNames = getFuncOpNames(builder, args.size(), "in");
  } else {
    llvm::transform(args, std::back_inserter(argNames), [&](auto arg) {
      return builder.getStringAttr(arg.ssaName.name.drop_front());
    });
  }

  // Parse attributes
  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();

  // If argNames and resNames wasn't provided manually, infer argNames attribute
  // from the parsed SSA names and resNames from our naming convention.
  if (!result.attributes.get("argNames"))
    result.addAttribute("argNames", builder.getArrayAttr(argNames));
  if (!result.attributes.get("resNames")) {
    auto resNames = getFuncOpNames(builder, resTypes.size(), "out");
    result.addAttribute("resNames", builder.getArrayAttr(resNames));
  }

  // Parse the optional function body. The printer will not print the body if
  // its empty, so disallow parsing of empty body in the parser.
  auto *body = result.addRegion();
  llvm::SMLoc loc = parser.getCurrentLocation();
  auto parseResult = parser.parseOptionalRegion(*body, args,
                                                /*enableNameShadowing=*/false);
  if (!parseResult.has_value())
    return success();

  if (failed(*parseResult))
    return failure();
  // Function body was parsed, make sure its not empty.
  if (body->empty())
    return parser.emitError(loc, "expected non-empty function body");

  // If a body was parsed, the arg and res names need to be resolved
  return success();
}

void FuncOp::print(OpAsmPrinter &p) {
  mlir::function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/true, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

bool ConditionalBranchOp::isControl() {
  return isControlCheckTypeAndOperand(getDataOperand().getType(),
                                      getDataOperand());
}

LogicalResult ConstantOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions,
    SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  OperationName opName = OperationName(getOperationName(), context);
  StringAttr attrName = getValueAttrName(opName);
  auto attr = cast<TypedAttr>(attributes.get(attrName));
  inferredReturnTypes.push_back(handshake::ChannelType::get(attr.getType()));
  return success();
}

LogicalResult ConstantOp::verify() {
  // Verify that the type of the provided value is equal to the result channel's
  // data type
  auto typedValue = dyn_cast<mlir::TypedAttr>(getValue());
  if (!typedValue)
    return emitOpError("constant value must be a typed attribute; value is ")
           << getValue();
  if (typedValue.getType() != getResult().getType().getDataType())
    return emitOpError() << "constant value type " << typedValue.getType()
                         << " differs from result channel's data type "
                         << getResult().getType();
  return success();
}

bool JoinOp::isControl() { return true; }

/// Based on mlir::func::CallOp::verifySymbolUses
LogicalResult InstanceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the module attribute was specified.
  auto fnAttr = this->getModuleAttr();
  assert(fnAttr && "requires a 'module' symbol reference attribute");

  FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid handshake function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError(
        "incorrect number of operands for the referenced handshake function");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != getNumResults())
    return emitOpError(
        "incorrect number of results for the referenced handshake function");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (getResult(i).getType() != fnType.getResult(i))
      return emitOpError("result type mismatch: expected result type ")
             << fnType.getResult(i) << ", but provided "
             << getResult(i).getType() << " for result number " << i;

  return success();
}

FunctionType InstanceOp::getModuleType() {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

/// Finds the type of operation that produces the input by backtracking the
/// def-use chain through potential bitwidth modification and/or fork
/// operations.
static Operation *backtrackToMemInput(Value input) {
  Operation *inputOp = input.getDefiningOp();
  while (isa_and_present<handshake::ExtSIOp, handshake::ExtUIOp,
                         handshake::TruncIOp, handshake::ForkOp>(inputOp))
    inputOp = inputOp->getOperand(0).getDefiningOp();
  return inputOp;
}

/// Common verification logic for memory controllers and LSQs.
static LogicalResult verifyMemOp(FuncMemoryPorts &ports) {
  // At most we can connect to a single other memory interface
  if (ports.interfacePorts.size() > 1)
    return ports.memOp.emitError() << "Interface may have at most one port to "
                                      "other memory interfaces, but has "
                                   << ports.interfacePorts.size() << ".";
  return success();
}

/// During construction of a memory interface's ports information, checks
/// whether bitwidths are consistent between all signals of the same nature
/// (e.g., address signals). A width of 0 is understood as "uninitialized", in
/// which case the first encountered signal determines the "reference" for that
/// signal type's width. Fails when the memory input's bitwidth doesn't match
/// the previously established bitwidth.
static LogicalResult checkAndSetBitwidth(Value memInput, unsigned &width) {
  // Determine the signal's width
  Type inType = memInput.getType();
  if (!isa<handshake::ControlType, handshake::ChannelType>(inType)) {
    return memInput.getUsers().begin()->emitError()
           << "Invalid input type, expected !handshake.control or "
              "!handshake.channel";
  }
  unsigned inputWidth = getHandshakeTypeBitWidth(inType);

  if (width == 0) {
    // This is the first time we encounter a signal of this type, consider its
    // width as the reference one for its category
    width = inputWidth;
    return success();
  }
  if (width != inputWidth) {
    return memInput.getUsers().begin()->emitError()
           << "Inconsistent bitwidths in inputs, expected signal with " << width
           << " bits but got signal with " << inputWidth << " bits.";
  }
  return success();
};

//===----------------------------------------------------------------------===//
// MemoryControllerOp
//===----------------------------------------------------------------------===//

static handshake::ChannelType wrapChannel(Type type) {
  assert(ChannelType::isSupportedSignalType(type) && "unsupported data type");
  return handshake::ChannelType::get(type);
}

void MemoryControllerOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                               Value memRef, Value memStart, ValueRange inputs,
                               Value ctrlEnd, ArrayRef<unsigned> blocks,
                               unsigned numLoads) {
  // Memory operands
  odsState.addOperands({memRef, memStart});
  odsState.addOperands(inputs);
  odsState.addOperands(ctrlEnd);

  // Data outputs (get their type from memref)
  MemRefType memrefType = memRef.getType().cast<MemRefType>();
  MLIRContext *ctx = odsBuilder.getContext();
  odsState.types.append(numLoads, wrapChannel(memrefType.getElementType()));
  odsState.types.push_back(handshake::ControlType::get(ctx));

  // Set "connectedBlocks" attribute
  SmallVector<int> blocksAttribute;
  for (unsigned size : blocks)
    blocksAttribute.push_back(size);
  odsState.addAttribute("connectedBlocks",
                        odsBuilder.getI32ArrayAttr(blocksAttribute));
}

/// Attempts to derive the ports for a memory controller. Fails if some of the
/// ports fails to be identified.
static LogicalResult getMCPorts(MCPorts &mcPorts) {
  unsigned nextBlockIdx = 0, resIdx = 0;
  std::optional<unsigned> currentBlockID;
  GroupMemoryPorts *currentGroup = nullptr;
  ValueRange memResults = mcPorts.memOp->getResults();
  SmallVector<unsigned> blocks = mcPorts.getMCOp().getMCBlocks();

  // Moves forward in our list of blocks, checking that we do not overflow it.
  auto getNextBlockID = [&]() -> LogicalResult {
    if (nextBlockIdx == blocks.size())
      return mcPorts.memOp.emitError()
             << "MC connects to more blocks than it declares.";
    currentBlockID = blocks[nextBlockIdx++];
    return success();
  };

  // Iterate over all memory inputs and figure out which ports are connected to
  // the memory controller
  auto inputIter = llvm::enumerate(mcPorts.memOp->getOperands());
  auto currentIt = inputIter.begin();

  // Skip the memref and memory start signals as well as the control end signal
  ++(++currentIt);
  unsigned lastIterOprd = mcPorts.memOp->getNumOperands() - 1;
  for (; currentIt != inputIter.end(); ++currentIt) {
    auto input = *currentIt;
    if (input.index() == lastIterOprd)
      break;
    Operation *portOp = backtrackToMemInput(input.value());

    // Identify the block the input operation belongs to
    unsigned portOpBlock = 0;
    if (portOp) {
      auto ctrlBlock =
          dyn_cast_if_present<IntegerAttr>(portOp->getAttr(BB_ATTR_NAME));
      if (ctrlBlock) {
        portOpBlock = ctrlBlock.getUInt();
      } else if (isa<handshake::MCLoadOp, handshake::MCStoreOp>(portOp)) {
        return portOp->emitError() << "Input operation of memory interface "
                                      "does not belong to any basic block.";
      }
    }

    // Checks wheter an access port belongs to the correct block
    auto checkAccessPort = [&]() -> LogicalResult {
      if (portOpBlock != *currentBlockID)
        return mcPorts.memOp->emitError()
               << "Access port belongs to block " << portOpBlock
               << ", but expected it to be part of block " << *currentBlockID;
      return success();
    };

    auto handleLoad = [&](handshake::MCLoadOp loadOp) -> LogicalResult {
      if (failed(checkAndSetBitwidth(input.value(), mcPorts.addrWidth)) ||
          failed(checkAndSetBitwidth(memResults[resIdx], mcPorts.dataWidth)))
        return failure();

      if (!currentGroup || portOpBlock != *currentBlockID) {
        // If this is the first input or if the load belongs to a different
        // block, allocate a new data stucture for the block's memory ports
        // (without control)
        if (failed(getNextBlockID()))
          return failure();
        mcPorts.groups.emplace_back();
        currentGroup = &mcPorts.groups.back();
      }

      if (failed(checkAccessPort()))
        return failure();

      // Add a load port to the group
      currentGroup->accessPorts.push_back(
          MCLoadPort(loadOp, input.index(), resIdx++));
      return success();
    };

    auto handleStore = [&](handshake::MCStoreOp storeOp) -> LogicalResult {
      auto dataInput = *(++currentIt);
      if (failed(checkAndSetBitwidth(input.value(), mcPorts.addrWidth)) ||
          failed(checkAndSetBitwidth(dataInput.value(), mcPorts.dataWidth)))
        return failure();

      // All basic blocks with at least one store access must have a control
      // input, so the block's memory ports must already be defined and point to
      // the same block
      if (!currentGroup)
        return storeOp.emitError() << "Store port must be preceeded by control "
                                      "port.";

      if (failed(checkAccessPort()))
        return failure();

      // Add a store port to the block
      currentGroup->accessPorts.push_back(MCStorePort(storeOp, input.index()));
      return success();
    };

    auto handleLSQ = [&](handshake::LSQOp lsqOp) -> LogicalResult {
      auto stAddrInput = *(++currentIt);
      auto stDataInput = *(++currentIt);
      if (failed(checkAndSetBitwidth(input.value(), mcPorts.addrWidth)) ||
          failed(checkAndSetBitwidth(memResults[resIdx], mcPorts.dataWidth)) ||
          failed(checkAndSetBitwidth(stAddrInput.value(), mcPorts.addrWidth)) ||
          failed(checkAndSetBitwidth(stDataInput.value(), mcPorts.dataWidth)))
        return failure();

      // Add the port to the list of ports from other memory
      // interfaces
      mcPorts.interfacePorts.push_back(
          LSQLoadStorePort(lsqOp, input.index(), resIdx++));
      return success();
    };

    auto handleControl = [&](Operation *ctrlOp) -> LogicalResult {
      if (failed(checkAndSetBitwidth(input.value(), mcPorts.ctrlWidth)) ||
          failed(getNextBlockID()))
        return failure();

      // Allocate a new data stucture for the group's memory ports
      mcPorts.groups.emplace_back(ControlPort(ctrlOp, input.index()));
      currentGroup = &mcPorts.groups.back();
      return success();
    };

    LogicalResult res = failure();
    if (!portOp) {
      // Control signal may come directly from function arguments
      res = handleControl(portOp);
    } else {
      res = llvm::TypeSwitch<Operation *, LogicalResult>(portOp)
                .Case<handshake::MCLoadOp>(handleLoad)
                .Case<handshake::MCStoreOp>(handleStore)
                .Case<handshake::LSQOp>(handleLSQ)
                .Default(handleControl);
    }

    // Forward failure when parsing parts
    if (failed(res))
      return failure();
  }

  // Check that all blocks have been accounted for
  if (nextBlockIdx != blocks.size())
    return mcPorts.memOp->emitError()
           << "MC declares more blocks than it connects to.";
  // Check that all memory results have been accounted for
  if (resIdx != memResults.size() - 1)
    return mcPorts.memOp->emitError()
           << "When identifying memory ports, some memory results were "
              "unnacounted for, this is not normal!";
  return success();
}

LogicalResult MemoryControllerOp::verify() {
  // Blocks must be unique
  SmallVector<unsigned> blocks = getMCBlocks();
  if (blocks.empty())
    return emitError() << "Memory controller must have at least one block.";
  DenseSet<unsigned> uniqueBlocks;
  for (unsigned blockID : blocks) {
    if (auto [_, newBlock] = uniqueBlocks.insert(blockID); !newBlock)
      return emitError() << "Block IDs must be unique but ID " << blockID
                         << " was found at least twice in the list.";
  }

  // Try to get the memort ports: this will catch most issues
  MCPorts mcPorts(*this);
  if (failed(getMCPorts(mcPorts)) || failed(verifyMemOp(mcPorts)))
    return failure();

  // If there is a port to another memory interface it must be to an LSQ
  if (!mcPorts.interfacePorts.empty()) {
    if (!isa<LSQLoadStorePort>(mcPorts.interfacePorts.front()))
      return emitError()
             << "The only memory interface port the memory controller "
                "supports is to an LSQ.";
  }
  return success();
}

dynamatic::MCPorts MemoryControllerOp::getPorts() {
  MCPorts mcPorts(*this);
  if (failed(getMCPorts(mcPorts)))
    assert(false && "failed to identify memory ports");
  return mcPorts;
}

//===----------------------------------------------------------------------===//
// LSQOp
//===----------------------------------------------------------------------===//

static void buildLSQGroupSizes(OpBuilder &odsBuilder, OperationState &odsState,
                               ArrayRef<unsigned> groupSizes) {
  SmallVector<int> sizesAttribute;
  for (unsigned size : groupSizes)
    sizesAttribute.push_back(size);
  odsState.addAttribute(LSQOp::getGroupSizesAttrName(odsState.name).strref(),
                        odsBuilder.getI32ArrayAttr(sizesAttribute));
}

void LSQOp::build(OpBuilder &odsBuilder, OperationState &odsState, Value memref,
                  Value memStart, ValueRange inputs, Value ctrlEnd,
                  ArrayRef<unsigned> groupSizes, unsigned numLoads) {
  // Memory operands
  odsState.addOperands({memref, memStart});
  odsState.addOperands(inputs);
  odsState.addOperands(ctrlEnd);

  // Data outputs (get their type from memref)
  MemRefType memrefType = memref.getType().cast<MemRefType>();
  MLIRContext *ctx = odsBuilder.getContext();
  odsState.types.append(numLoads, wrapChannel(memrefType.getElementType()));
  odsState.types.push_back(handshake::ControlType::get(ctx));
  buildLSQGroupSizes(odsBuilder, odsState, groupSizes);
}

void LSQOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                  handshake::MemoryControllerOp mcOp, ValueRange inputs,
                  ArrayRef<unsigned> groupSizes, unsigned numLoads) {
  // Memory operands
  odsState.addOperands(inputs);

  // Data outputs (get their type from memref)
  MemRefType memrefType = mcOp.getMemRefType();
  MLIRContext *ctx = odsBuilder.getContext();
  Type dataType = wrapChannel(memrefType.getElementType());
  odsState.types.append(numLoads, dataType);

  // Add results for load/store address and store data
  Type addrType = handshake::ChannelType::getAddrChannel(ctx);
  odsState.types.append(2, addrType);
  odsState.types.push_back(dataType);

  // The LSQ is a slave interface in this case (the MC is the master), so it
  // doesn't produce a completion signal

  buildLSQGroupSizes(odsBuilder, odsState, groupSizes);
}

ParseResult LSQOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<Type> resultTypes, operandTypes;
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();

  // Parse reference to memory region
  if (parser.parseLSquare())
    return failure();
  if (parser.parseOptionalKeyword("MC")) {
    // We expect something of the form "memref operand : memref type"
    OpAsmParser::UnresolvedOperand memrefOperand;
    Type memrefType;
    if (parser.parseOperand(memrefOperand) || parser.parseColon() ||
        parser.parseType(memrefType))
      return failure();
    operands.push_back(memrefOperand);
    operandTypes.push_back(memrefType);
  }
  if (parser.parseRSquare())
    return failure();

  // Parse memory operands
  if (parser.parseLParen() || parser.parseOperandList(operands) ||
      parser.parseRParen())
    return failure();

  // Parse group sizes and other attributes
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse operand and result types (function type)
  FunctionType funType;
  if (parser.parseColonType(funType))
    return failure();
  llvm::copy(funType.getInputs(), std::back_inserter(operandTypes));
  llvm::copy(funType.getResults(), std::back_inserter(resultTypes));

  // Set result types and resolve operand types
  result.addTypes(resultTypes);
  if (parser.resolveOperands(operands, operandTypes, allOperandLoc,
                             result.operands))
    return failure();
  return success();
}

void LSQOp::print(OpAsmPrinter &p) {
  // Print reference to memory region
  bool connectsToMC = isConnectedToMC();
  if (connectsToMC) {
    p << "[MC] ";
  } else {
    Value memref = getInputs().front();
    p << "[" << memref << " : " << memref.getType() << "] ";
  }

  // Print memory operands
  ValueRange inputs = getOperands();
  if (!connectsToMC) {
    // Skip the memref if present
    inputs = inputs.drop_front();
  }
  p << "(";
  for (Value oprd : inputs.drop_back(1))
    p << oprd << ", ";
  p << inputs.back() << ") ";

  // Print group sizes and other attributes
  p.printOptionalAttrDict((*this)->getAttrs());

  // Print result types
  p << " : ";
  p.printFunctionalType(inputs.getTypes(), getResults().getTypes());
}

/// Attempts to derive the ports for an LSQ. Fails if some of the ports fails to
/// be identified.
static LogicalResult getLSQPorts(LSQPorts &lsqPorts) {
  unsigned nextGroupIdx = 0, resIdx = 0;
  std::optional<unsigned> currentGroupRemaining;
  GroupMemoryPorts *currentGroup = nullptr;
  ValueRange memResults = lsqPorts.memOp->getResults();
  SmallVector<unsigned> groupSizes = lsqPorts.getLSQOp().getLSQGroupSizes();

  auto checkGroupIsValid = [&]() -> LogicalResult {
    if (!currentGroupRemaining)
      return lsqPorts.memOp->emitError()
             << "Access port must be precedeed by control port to start group";
    if (*currentGroupRemaining == 0)
      return lsqPorts.memOp->emitError()
             << "There are more access ports in group " << nextGroupIdx - 1
             << " than what the LSQ declares.";
    return success();
  };

  // Iterate over all memory inputs and figure out which ports are connected to
  // the memory controller
  auto inputIter = llvm::enumerate(lsqPorts.memOp->getOperands());
  auto currentIt = inputIter.begin();
  auto iterEnd = inputIter.end();

  unsigned lastIterOprd = lsqPorts.memOp->getNumOperands();
  if (lsqPorts.memOp.isMasterInterface()) {
    // Skip the memref and memory start signals as well as the ctrl end signal
    ++(++currentIt);
    --lastIterOprd;
  }
  for (; currentIt != iterEnd; ++currentIt) {
    auto input = *currentIt;
    if (input.index() == lastIterOprd)
      break;

    auto handleLoad = [&](handshake::LSQLoadOp loadOp) -> LogicalResult {
      if (failed(checkAndSetBitwidth(input.value(), lsqPorts.addrWidth)) ||
          failed(checkAndSetBitwidth(memResults[resIdx], lsqPorts.dataWidth)) ||
          failed(checkGroupIsValid()))
        return failure();

      // Add a load port to the group
      currentGroup->accessPorts.push_back(
          LSQLoadPort(loadOp, input.index(), resIdx++));
      --(*currentGroupRemaining);
      return success();
    };

    auto handleStore = [&](handshake::LSQStoreOp storeOp) -> LogicalResult {
      auto dataInput = *(++currentIt);
      if (failed(checkAndSetBitwidth(input.value(), lsqPorts.addrWidth)) ||
          failed(checkAndSetBitwidth(dataInput.value(), lsqPorts.dataWidth)) ||
          failed(checkGroupIsValid()))
        return failure();

      // Add a store port to the group and decrement our group size by one
      currentGroup->accessPorts.push_back(LSQStorePort(storeOp, input.index()));
      --(*currentGroupRemaining);
      return success();
    };

    auto handleMC = [&](handshake::MemoryControllerOp mcOp) -> LogicalResult {
      if (failed(checkAndSetBitwidth(input.value(), lsqPorts.dataWidth)) ||
          failed(checkAndSetBitwidth(memResults[resIdx], lsqPorts.addrWidth)) ||
          failed(checkAndSetBitwidth(memResults[resIdx + 1],
                                     lsqPorts.addrWidth)) ||
          failed(
              checkAndSetBitwidth(memResults[resIdx + 2], lsqPorts.dataWidth)))
        return failure();

      // Add the port to the list of ports from other memory interfaces
      lsqPorts.interfacePorts.push_back(
          MCLoadStorePort(mcOp, resIdx, input.index()));
      resIdx += 3;
      return success();
    };

    auto handleControl = [&](Operation *ctrlOp) -> LogicalResult {
      if (failed(checkAndSetBitwidth(input.value(), lsqPorts.ctrlWidth)))
        return failure();

      // Moves forward in our list of groups, checking that we do not overflow
      // it and that the previous one (if any) was completed
      if (nextGroupIdx == groupSizes.size())
        return lsqPorts.memOp.emitError()
               << "LSQ has more groups than it declares.";
      if (currentGroupRemaining && *currentGroupRemaining != 0)
        return lsqPorts.memOp.emitError()
               << "Group " << nextGroupIdx - 1 << " is missing "
               << *currentGroupRemaining
               << " access ports compared to what the LSQ declares.";
      currentGroupRemaining = groupSizes[nextGroupIdx++];

      // Allocate a new data stucture for the group's memory ports
      lsqPorts.groups.emplace_back(ControlPort(ctrlOp, input.index()));
      currentGroup = &lsqPorts.groups.back();
      return success();
    };

    Operation *portOp = backtrackToMemInput(input.value());
    LogicalResult res = failure();
    if (!portOp) {
      // Control signal may come directly from function arguments
      res = handleControl(portOp);
    } else {
      res = llvm::TypeSwitch<Operation *, LogicalResult>(portOp)
                .Case<handshake::LSQLoadOp>(handleLoad)
                .Case<handshake::LSQStoreOp>(handleStore)
                .Case<handshake::MemoryControllerOp>(handleMC)
                .Default(handleControl);
    }

    // Forward failure when parsing parts
    if (failed(res))
      return failure();
  }

  // Check that all groups have been accounted for
  if (nextGroupIdx != groupSizes.size())
    return lsqPorts.memOp->emitError()
           << "LSQ declares more groups than it connects to.";
  // Check that all memory results have been accounted for
  unsigned expectedResIdx = memResults.size();
  if (lsqPorts.memOp.isMasterInterface())
    expectedResIdx -= 1;
  if (resIdx != expectedResIdx) {
    return lsqPorts.memOp->emitError()
           << "Some memory results were unnacounted for when identifying ports";
  }
  return success();
}

LogicalResult LSQOp::verify() {
  // Group sizes must be strictly greater than 0
  SmallVector<unsigned> sizes = getLSQGroupSizes();
  if (sizes.empty())
    return emitError() << "LSQ needs to have at least one group.";
  for (unsigned size : sizes) {
    if (size == 0)
      return emitError() << "Group sizes must be stricly greater than 0.";
  }

  // Try to get the memort ports: this will catch most issues
  LSQPorts lsqPorts(*this);
  if (failed(getLSQPorts(lsqPorts)) || failed(verifyMemOp(lsqPorts)))
    return failure();

  // If there is a port to another memory interface it must be to an LSQ
  if (!lsqPorts.interfacePorts.empty()) {
    if (!isa<MCLoadStorePort>(lsqPorts.interfacePorts.front()))
      return emitError() << "The only memory interface port the LSQ supports "
                            "is to a memory controller.";
  }
  return success();
}

dynamatic::LSQPorts LSQOp::getPorts() {
  LSQPorts lsqPorts(*this);
  if (failed(getLSQPorts(lsqPorts)))
    assert(false && "failed to identify memory ports");
  return lsqPorts;
}

handshake::MemoryControllerOp LSQOp::getConnectedMC() {
  auto storeDataUsers = getResults().back().getUsers();
  if (storeDataUsers.empty())
    return nullptr;
  return dyn_cast_if_present<dynamatic::handshake::MemoryControllerOp>(
      *storeDataUsers.begin());
}

//===----------------------------------------------------------------------===//
// Specific port kinds
//===----------------------------------------------------------------------===//

dynamatic::FuncMemoryPorts
dynamatic::getMemoryPorts(handshake::MemoryOpInterface memOp) {
  if (auto mcOp = dyn_cast<handshake::MemoryControllerOp>((Operation *)memOp)) {
    MCPorts mcPorts(mcOp);
    if (failed(getMCPorts(mcPorts)))
      assert(false && "failed to identify memory ports");
    return mcPorts;
  }
  if (auto lsqOp = dyn_cast<handshake::LSQOp>((Operation *)memOp)) {
    LSQPorts lsqPorts(lsqOp);
    if (failed(getLSQPorts(lsqPorts)))
      assert(false && "failed to identify memory ports");
    return lsqPorts;
  }
  llvm_unreachable("unknown memory interface type");
}

// MemoryPort (asbtract)

MemoryPort::MemoryPort(Operation *portOp, ArrayRef<unsigned> oprdIndices,
                       ArrayRef<unsigned> resIndices, Kind kind)
    : portOp(portOp), oprdIndices(oprdIndices), resIndices(resIndices),
      kind(kind) {}

// ControlPort: op -> mem. interface

ControlPort::ControlPort(Operation *ctrlOp, unsigned ctrlInputIdx)
    : MemoryPort(ctrlOp, {ctrlInputIdx}, {}, Kind::CONTROL) {}

// LoadPort: LoadOpInterface <-> mem. interface

LoadPort::LoadPort(handshake::LoadOpInterface loadOp, unsigned addrInputIdx,
                   unsigned dataOutputIdx, Kind kind)
    : MemoryPort(loadOp, {addrInputIdx}, {dataOutputIdx}, kind) {}

handshake::LoadOpInterface LoadPort::getLoadOp() const {
  return cast<handshake::LoadOpInterface>(portOp);
}

MCLoadPort::MCLoadPort(handshake::MCLoadOp loadOp, unsigned addrInputIdx,
                       unsigned dataOutputIdx)
    : LoadPort(loadOp, addrInputIdx, dataOutputIdx, Kind::MC_LOAD) {}

handshake::MCLoadOp MCLoadPort::getMCLoadOp() const {
  return cast<handshake::MCLoadOp>(portOp);
}

LSQLoadPort::LSQLoadPort(handshake::LSQLoadOp loadOp, unsigned addrInputIdx,
                         unsigned dataOutputIdx)
    : LoadPort(loadOp, addrInputIdx, dataOutputIdx, Kind::LSQ_LOAD) {}

handshake::LSQLoadOp LSQLoadPort::getLSQLoadOp() const {
  return cast<handshake::LSQLoadOp>(portOp);
}

// StorePort: StoreOpInterface -> mem. interface

StorePort::StorePort(handshake::StoreOpInterface storeOp, unsigned addrInputIdx,
                     Kind kind)
    : MemoryPort(storeOp, {addrInputIdx, addrInputIdx + 1}, {}, kind) {};

handshake::StoreOpInterface StorePort::getStoreOp() const {
  return cast<handshake::StoreOpInterface>(portOp);
}

MCStorePort::MCStorePort(handshake::MCStoreOp storeOp, unsigned addrInputIdx)
    : StorePort(storeOp, addrInputIdx, Kind::MC_STORE) {}

handshake::MCStoreOp MCStorePort::getMCStoreOp() const {
  return cast<handshake::MCStoreOp>(portOp);
}

LSQStorePort::LSQStorePort(handshake::LSQStoreOp storeOp, unsigned addrInputIdx)
    : StorePort(storeOp, addrInputIdx, Kind::LSQ_STORE) {}

handshake::LSQStoreOp LSQStorePort::getLSQStoreOp() const {
  return cast<handshake::LSQStoreOp>(portOp);
}

// LSQLoadStorePort: LSQOp <-> mem. interface

LSQLoadStorePort::LSQLoadStorePort(dynamatic::handshake::LSQOp lsqOp,
                                   unsigned loadAddrInputIdx,
                                   unsigned loadDataOutputIdx)
    : MemoryPort(lsqOp,
                 {loadAddrInputIdx, loadAddrInputIdx + 1, loadAddrInputIdx + 2},
                 {loadDataOutputIdx}, Kind::LSQ_LOAD_STORE) {}

handshake::LSQOp LSQLoadStorePort::getLSQOp() const {
  return cast<handshake::LSQOp>(portOp);
}

// LSQLoadStorePort: LSQOp <-> mem. interface

MCLoadStorePort::MCLoadStorePort(dynamatic::handshake::MemoryControllerOp mcOp,
                                 unsigned loadAddrOutputIdx,
                                 unsigned loadDataInputIdx)
    : MemoryPort(
          mcOp, {loadDataInputIdx},
          {loadAddrOutputIdx, loadAddrOutputIdx + 1, loadAddrOutputIdx + 2},
          Kind::MC_LOAD_STORE) {}

handshake::MemoryControllerOp MCLoadStorePort::getMCOp() const {
  return cast<handshake::MemoryControllerOp>(portOp);
}

//===----------------------------------------------------------------------===//
// GroupMemoryPorts
//===----------------------------------------------------------------------===//

GroupMemoryPorts::GroupMemoryPorts(ControlPort ctrlPort)
    : ctrlPort(ctrlPort) {};

unsigned GroupMemoryPorts::getNumInputs() const {
  unsigned numInputs = hasControl() ? 1 : 0;
  for (const MemoryPort &port : accessPorts) {
    if (isa<LoadPort>(port))
      numInputs += 1;
    else if (isa<StorePort>(port))
      numInputs += 2;
  }
  return numInputs;
}

unsigned GroupMemoryPorts::getNumResults() const {
  unsigned numResults = 0;
  for (const MemoryPort &port : accessPorts) {
    // There is one data output per load port
    if (isa<LoadPort>(port))
      numResults += 1;
  }
  return numResults;
}

size_t GroupMemoryPorts::getFirstOperandIndex() const {
  if (ctrlPort)
    return (*ctrlPort).getCtrlInputIndex();
  for (const MemoryPort &port : accessPorts) {
    if (auto indices = port.getOprdIndices(); !indices.empty())
      return indices.front();
  }
  return std::string::npos;
}

size_t GroupMemoryPorts::getLastOperandIndex() const {
  for (const MemoryPort &port : llvm::reverse(accessPorts)) {
    if (auto indices = port.getOprdIndices(); !indices.empty())
      return indices.back();
  }
  if (ctrlPort)
    return (*ctrlPort).getCtrlInputIndex();
  return std::string::npos;
}

size_t GroupMemoryPorts::getFirstResultIndex() const {
  for (const MemoryPort &port : accessPorts) {
    if (auto indices = port.getResIndices(); !indices.empty())
      return indices.front();
  }
  return std::string::npos;
}

size_t GroupMemoryPorts::getLastResultIndex() const {
  for (const MemoryPort &port : llvm::reverse(accessPorts)) {
    if (auto indices = port.getResIndices(); !indices.empty())
      return indices.front();
  }
  return std::string::npos;
}

//===----------------------------------------------------------------------===//
// FuncMemoryPorts (and children)
//===----------------------------------------------------------------------===//

mlir::ValueRange FuncMemoryPorts::getGroupInputs(unsigned groupIdx) {
  assert(groupIdx < groups.size() && "index higher than number of groups");
  size_t firstIdx = groups[groupIdx].getFirstOperandIndex();
  if (firstIdx == std::string::npos)
    return {};
  size_t lastIdx = groups[groupIdx].getLastOperandIndex();
  return memOp->getOperands().slice(firstIdx, lastIdx - firstIdx + 1);
}

mlir::ValueRange FuncMemoryPorts::getGroupResults(unsigned groupIdx) {
  assert(groupIdx < groups.size() && "index higher than number of groups");
  size_t firstIdx = groups[groupIdx].getFirstResultIndex();
  if (firstIdx == std::string::npos)
    return {};
  size_t lastIdx = groups[groupIdx].getLastResultIndex();
  return memOp->getResults().slice(firstIdx, lastIdx - firstIdx + 1);
}

mlir::ValueRange FuncMemoryPorts::getInterfacesInputs() {
  ValueRange memResults = memOp->getOperands();
  for (const MemoryPort &port : interfacePorts) {
    if (std::optional<MCLoadStorePort> mc = dyn_cast<MCLoadStorePort>(port)) {
      return memResults.drop_front(mc->getLoadDataInputIndex());
    }
    std::optional<LSQLoadStorePort> lsq = dyn_cast<LSQLoadStorePort>(port);
    assert(lsq && "port must be mc load/store or lsq load/store");
    return memResults.drop_front(lsq->getLoadAddrInputIndex());
  }
  return {};
}

mlir::ValueRange FuncMemoryPorts::getInterfacesResults() {
  ValueRange memOperands = memOp->getResults();
  for (const MemoryPort &port : interfacePorts) {
    if (std::optional<MCLoadStorePort> mc = dyn_cast<MCLoadStorePort>(port)) {
      return memOperands.drop_front(mc->getLoadAddrOutputIndex());
    }
    std::optional<LSQLoadStorePort> lsq = dyn_cast<LSQLoadStorePort>(port);
    assert(lsq && "port must be mc load/store or lsq load/store");
    return memOperands.drop_front(lsq->getLoadDataOutputIndex());
  }
  return {};
}

MCBlock::MCBlock(GroupMemoryPorts *group, unsigned blockID)
    : blockID(blockID), group(group) {};

MCPorts::MCPorts(handshake::MemoryControllerOp mcOp) : FuncMemoryPorts(mcOp) {};

handshake::MemoryControllerOp MCPorts::getMCOp() const {
  return cast<handshake::MemoryControllerOp>(memOp);
}

MCBlock MCPorts::getBlock(unsigned blockIdx) {
  return MCBlock(&groups[blockIdx], getMCOp().getMCBlocks()[blockIdx]);
}

SmallVector<MCBlock> MCPorts::getBlocks() {
  SmallVector<MCBlock> mcBlocks;
  SmallVector<unsigned> blockIDs = getMCOp().getMCBlocks();
  for (auto [ports, id] : llvm::zip(groups, blockIDs))
    mcBlocks.emplace_back(&ports, id);
  return mcBlocks;
}

LSQLoadStorePort MCPorts::getLSQPort() const {
  assert(hasConnectionToLSQ() && "no LSQ connected");
  std::optional<LSQLoadStorePort> lsqPort =
      dyn_cast<LSQLoadStorePort>(interfacePorts.front());
  assert(lsqPort && "lsq load/store port undefined");
  return *lsqPort;
}

LSQGroup::LSQGroup(GroupMemoryPorts *group) : group(group) {};

SmallVector<LSQGroup> LSQPorts::getGroups() {
  SmallVector<LSQGroup> lsqGroups;
  for (GroupMemoryPorts &ports : groups)
    lsqGroups.emplace_back(&ports);
  return lsqGroups;
}

LSQPorts::LSQPorts(handshake::LSQOp lsqOp) : FuncMemoryPorts(lsqOp) {};

handshake::LSQOp LSQPorts::getLSQOp() const {
  return cast<handshake::LSQOp>(memOp);
}

MCLoadStorePort LSQPorts::getMCPort() const {
  assert(hasConnectionToMC() && "no LSQ connected");
  std::optional<MCLoadStorePort> mcPort =
      dyn_cast<MCLoadStorePort>(interfacePorts.front());
  assert(mcPort && "mc load/store port undefined");
  return *mcPort;
}

//===----------------------------------------------------------------------===//
// MCLoadOp/LSQLoadOp
//===----------------------------------------------------------------------===//

/// Common builder for load ports that only adds the address operand (the data
/// operand should be added manually later).
static void buildLoadPort(OperationState &odsState, MemRefType memrefType,
                          Value address) {
  // Address (data value will be added later in the elastic pass)
  odsState.addOperands(address);

  // Data and address outputs
  odsState.types.push_back(address.getType());
  odsState.types.push_back(wrapChannel(memrefType.getElementType()));
}

void MCLoadOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                     MemRefType memrefType, Value address) {
  buildLoadPort(odsState, memrefType, address);
}

void LSQLoadOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                      MemRefType memrefType, Value address) {
  buildLoadPort(odsState, memrefType, address);
}

//===----------------------------------------------------------------------===//
// EndOp
//===----------------------------------------------------------------------===//

LogicalResult EndOp::verify() {
  // Number of operands must match number of function results
  TypeRange oprdTypes = getOperandTypes();
  TypeRange resTypes = getParentOp().getFunctionType().getResults();
  if (oprdTypes.size() != resTypes.size()) {
    return emitError() << "number of operands must match number of "
                          "function results, expected "
                       << resTypes.size() << " but got " << oprdTypes.size();
  }

  // Operand types must match function result types
  for (auto [idx, types] :
       llvm::enumerate(llvm::zip_equal(oprdTypes, resTypes))) {
    auto &[oprd, res] = types;
    if (oprd != res) {
      return emitError() << "operand " << idx << "'s type must match result "
                         << idx << "'s type, expected " << res << " but got "
                         << oprd;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SpeculatorOp
//===----------------------------------------------------------------------===//

ParseResult SpeculatorOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand enable, dataIn;
  Type dataType;
  SmallVector<Type> uniqueResTypes;
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseLSquare() || parser.parseOperand(enable) ||
      parser.parseRSquare() || parser.parseOperand(dataIn) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(dataType) || parser.parseArrowTypeList(uniqueResTypes))
    return failure();
  if (uniqueResTypes.size() != 3)
    return failure();

  Type ctrlType = uniqueResTypes[1];
  Type wideCtrlType = uniqueResTypes[2];
  result.addTypes(
      {dataType, ctrlType, ctrlType, wideCtrlType, wideCtrlType, ctrlType});

  if (parser.resolveOperands({enable, dataIn},
                             {ControlType::get(parser.getContext()), dataType},
                             allOperandLoc, result.operands))
    return failure();
  return success();
}

void SpeculatorOp::print(OpAsmPrinter &p) {
  p << " [" << getEnable() << "] " << getDataIn() << " ";
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << getDataIn().getType() << " -> (" << getDataOut().getType()
    << ", " << getSaveCtrl().getType() << ", " << getSCSaveCtrl().getType()
    << ")";
}

LogicalResult SpeculatorOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions,
    SmallVectorImpl<mlir::Type> &inferredReturnTypes) {

  inferredReturnTypes.push_back(operands.front().getType());

  OpBuilder builder(context);
  ChannelType ctrlType = ChannelType::get(builder.getIntegerType(1));
  ChannelType wideControlType = ChannelType::get(builder.getIntegerType(3));
  inferredReturnTypes.push_back(ctrlType);
  inferredReturnTypes.push_back(ctrlType);
  inferredReturnTypes.push_back(wideControlType);
  inferredReturnTypes.push_back(wideControlType);
  inferredReturnTypes.push_back(ctrlType);
  return success();
}

LogicalResult SpeculatorOp::verify() {
  Type refBoolType = getSaveCtrl().getType();
  if (refBoolType != getCommitCtrl().getType() ||
      refBoolType != getSCBranchCtrl().getType()) {
    return emitError() << "expected $saveCtrl, $commitCtrl, and $SCBranchCtrl "
                          "to have the same type, but got "
                       << refBoolType << ", " << getCommitCtrl().getType()
                       << ", and " << getSCBranchCtrl().getType();
  }

  if (getSCSaveCtrl().getType() != getSCCommitCtrl().getType()) {
    return emitError() << "expected $SCSaveCtrl and $SCCommitCtrl to have the "
                          "same type, but got "
                       << getSCSaveCtrl().getType() << " and "
                       << getSCCommitCtrl().getType();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// BundleOp
//===----------------------------------------------------------------------===//

ParseResult BundleOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> signals;
  Type dataflowType;
  MLIRContext *ctx = parser.getContext();
  llvm::SMLoc operandLoc = parser.getCurrentLocation();

  if (parser.parseOperandList(signals) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseKeyword("_") || parser.parseKeyword("to") ||
      parseHandshakeType(parser, dataflowType))
    return failure();
  result.addTypes(dataflowType);

  // Determine the type of all input signals and of all upstream signals based
  // on the first result's type
  SmallVector<Type, 4> signalTypes;
  llvm::TypeSwitch<Type, void>(dataflowType)
      .Case<ChannelType>([&](ChannelType channelType) {
        signalTypes.push_back(ControlType::get(ctx));
        signalTypes.push_back(channelType.getDataType());
        for (const ExtraSignal &extra : channelType.getExtraSignals()) {
          if (extra.downstream)
            signalTypes.push_back(extra.type);
          else
            result.addTypes(extra.type);
        }
      })
      .Case<ControlType>([&](auto) {
        IntegerType i1Type = IntegerType::get(ctx, 1);
        signalTypes.push_back(i1Type);
        result.addTypes(i1Type);
      });

  return parser.resolveOperands(signals, signalTypes, operandLoc,
                                result.operands);
}

void BundleOp::print(OpAsmPrinter &p) {
  p << " " << getOperands() << " ";
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : _ to ";
  printHandshakeType(p, getChannelLike().getType());
}

/// Common verifier for `handshake::BundleOp` and `handshake::UnbundleOp`
/// operations, which have the same semantics. Emits an error and fails if the
/// operation under verification has inconsistent operands/results; succeeds
/// otherwise.
static LogicalResult
verifyBundlingOp(Type channelLikeType, ValueRange upstreams,
                 ValueRange downstreams, bool isBundleOp,
                 function_ref<InFlightDiagnostic()> emitError) {
  // Define some keyword for error reporting based on whether the operation is a
  // bundle or unbundle
  StringRef downStr, upStr, actionStr;
  if (isBundleOp) {
    downStr = "operand";
    upStr = "result";
    actionStr = "bundling";
  } else {
    downStr = "result";
    upStr = "operand";
    actionStr = "unbundling";
  }

  if (auto channelType = dyn_cast<handshake::ChannelType>(channelLikeType)) {
    // The first and second downstream values must be the channel's control and
    // the data type, respectively
    if (downstreams.size() < 2) {
      return emitError() << "not enough " << downStr << "s, " << actionStr
                         << " a !handshake.channel should "
                            "produce at least two "
                         << downStr << "s";
    }
    if (!isa<handshake::ControlType>(downstreams.front().getType())) {
      return emitError()
             << "type mistmatch between expected !handshake.control type and "
                "operation's first "
             << downStr << " (" << downstreams.front().getType() << ")";
    }
    downstreams = downstreams.drop_front();
    if (channelType.getDataType() != downstreams.front().getType()) {
      return emitError() << "type mismatch between channel's data type ("
                         << channelType.getDataType()
                         << ") and operation's second " << downStr << " ("
                         << downstreams.front().getType() << ")";
    }
    downstreams = downstreams.drop_front();

    // The channel's extra signals must be reflected in number and types in the
    // operation's operands and results depending on their direction
    auto downstreamIt = downstreams.begin();
    auto upstreamIt = upstreams.begin();
    for (const ExtraSignal &extra : channelType.getExtraSignals()) {
      auto handleExtraSignal = [&](auto &it, ValueRange values,
                                   StringRef valueType,
                                   unsigned valOffset) -> LogicalResult {
        if (it == values.end()) {
          return emitError()
                 << "not enough " << valueType
                 << "s, no value for extra signal '" << extra.name << "'";
        }
        if (extra.type != (*it).getType()) {
          unsigned valIdx = std::distance(values.begin(), it) + valOffset;
          return emitError()
                 << "type mismatch between extra signal '" << extra.name
                 << "' (" << extra.type << ") and " << valIdx << "-th "
                 << valueType << " (" << (*it).getType() << ")";
        }
        ++it;
        return success();
      };

      if (extra.downstream) {
        if (failed(handleExtraSignal(downstreamIt, downstreams, downStr, 2)))
          return failure();
      } else {
        if (failed(handleExtraSignal(upstreamIt, upstreams, upStr, 1)))
          return failure();
      }
    }
    if (downstreamIt != downstreams.end()) {
      return emitError()
             << "too many extra downstream values provided, expected "
             << std::distance(downstreams.begin(), downstreamIt) << " but got "
             << downstreams.size();
    }
    if (upstreamIt != upstreams.end()) {
      return emitError() << "too many extra upstream values provided, expected "
                         << std::distance(upstreams.begin(), upstreamIt)
                         << " but got " << upstreams.size();
    }
  } else {
    assert(isa<handshake::ControlType>(channelLikeType) && "expected control");
    OpBuilder builder(channelLikeType.getContext());
    IntegerType i1Type = builder.getIntegerType(1);
    if (upstreams.size() != 1 || upstreams.front().getType() != i1Type)
      return emitError() << "expected single i1 " << upStr << " for ready";
    if (downstreams.size() != 1 || downstreams.front().getType() != i1Type)
      return emitError() << "expected single i1 " << downStr << " for valid";
  }

  return success();
}

void BundleOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                     Value ctrl, Value data, ValueRange downstreams,
                     ChannelType channelType) {
  assert(isa<handshake::ControlType>(ctrl.getType()) &&
         "expected !handshake.control");
  assert(handshake::ChannelType::isSupportedSignalType(data.getType()) &&
         "unsupported data type");
  assert(downstreams.size() == channelType.getNumDownstreamExtraSignals() &&
         "incorrect number of extra downstream signals");

  odsState.addOperands({ctrl, data});
  odsState.addOperands(downstreams);

  // The operation produces the bundled channel type as well as any upstream
  // signal in the channel separately (in the order in which they are declared
  // by the channel)
  odsState.addTypes(channelType);
  for (const ExtraSignal &extra : channelType.getExtraSignals()) {
    if (!extra.downstream)
      odsState.addTypes(extra.type);
  }
}

LogicalResult BundleOp::verify() {
  return verifyBundlingOp(getChannelLike().getType(), getUpstreams(),
                          getSignals(), true, [&]() { return emitError(); });
}

//===----------------------------------------------------------------------===//
// UnbundleOp
//===----------------------------------------------------------------------===//

ParseResult UnbundleOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  OpAsmParser::UnresolvedOperand &channelLike = operands.emplace_back();
  Type dataflowType;
  MLIRContext *ctx = parser.getContext();
  llvm::SMLoc operandLoc = parser.getCurrentLocation();

  // Parse operands
  if (parser.parseOperand(channelLike))
    return failure();
  if (!parser.parseOptionalLSquare()) {
    if (parser.parseOperandList(operands) || parser.parseRSquare())
      return failure();
  }

  // Parse attribute dictionarry and type
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parseHandshakeType(parser, dataflowType) || parser.parseKeyword("to") ||
      parser.parseKeyword("_"))
    return failure();

  // Determine the type of all output signals and of all upstream signals based
  // on the first operand's type
  SmallVector<Type, 4> operandTypes;
  operandTypes.push_back(dataflowType);
  llvm::TypeSwitch<Type, void>(dataflowType)
      .Case<ChannelType>([&](ChannelType channelType) {
        result.addTypes({ControlType::get(ctx), channelType.getDataType()});
        for (const ExtraSignal &extra : channelType.getExtraSignals()) {
          if (extra.downstream)
            result.addTypes(extra.type);
          else
            operandTypes.push_back(extra.type);
        }
      })
      .Case<ControlType>([&](auto) {
        IntegerType i1Type = IntegerType::get(ctx, 1);
        result.addTypes(i1Type);
        operandTypes.push_back(i1Type);
      });

  return parser.resolveOperands(operands, operandTypes, operandLoc,
                                result.operands);
}

void UnbundleOp::print(OpAsmPrinter &p) {
  p << " " << getChannelLike() << " ";
  if (OperandRange upstreams = getUpstreams(); !upstreams.empty())
    p << "[ " << upstreams << " ] ";
  p.printOptionalAttrDict((*this)->getAttrs());
  printHandshakeType(p, getChannelLike().getType());
  p << " : to _ ";
}

LogicalResult UnbundleOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions,
    SmallVectorImpl<mlir::Type> &inferredReturnTypes) {

  Type channelLikeType = operands.front().getType();
  if (auto channelType = dyn_cast<handshake::ChannelType>(channelLikeType)) {
    inferredReturnTypes.push_back(handshake::ControlType::get(context));
    inferredReturnTypes.push_back(channelType.getDataType());
    // All downstream extra signals get unbundled after the control/data
    for (const ExtraSignal &extra : channelType.getExtraSignals()) {
      if (extra.downstream)
        inferredReturnTypes.push_back(extra.type);
    }
  } else {
    assert(isa<handshake::ControlType>(channelLikeType) && "expected control");
    OpBuilder builder(context);
    inferredReturnTypes.push_back(builder.getIntegerType(1));
  }

  return success();
}

LogicalResult UnbundleOp::verify() {
  return verifyBundlingOp(getChannelLike().getType(), getUpstreams(),
                          getSignals(), false, [&]() { return emitError(); });
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

/// Returns the total number of bits needed to carry all extra downstream
/// signals and all extra upstream signals.
static std::pair<unsigned, unsigned> getTotalExtraWidths(ChannelType type) {
  unsigned totalDownWidth = 0, totalUpWidth = 0;
  for (const ExtraSignal &extra : type.getExtraSignals()) {
    if (extra.downstream)
      totalDownWidth += extra.getBitWidth();
    else
      totalUpWidth += extra.getBitWidth();
  }
  return {totalDownWidth, totalUpWidth};
}

void ReshapeOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                      TypedValue<ChannelType> channel,
                      bool mergeDownstreamIntoData) {
  // Set the operands/attributes
  MLIRContext *ctx = odsBuilder.getContext();
  ChannelReshapeType reshapeType = mergeDownstreamIntoData
                                       ? ChannelReshapeType::MergeData
                                       : ChannelReshapeType::MergeExtra;
  odsState.addAttribute("reshapeType",
                        ChannelReshapeTypeAttr::get(ctx, reshapeType));
  odsState.addOperands(channel);

  // All extra signals will be combined into at most one downstream combined
  // signal and one upstream combined signal, compute their bitwidth
  auto [totalDownWidth, totalUpWidth] = getTotalExtraWidths(channel.getType());

  // The result channel type can be determined from the input channel type and
  // reshaping type
  SmallVector<ExtraSignal> extraSignals;
  Type dataType = channel.getType().getDataType();
  if (mergeDownstreamIntoData) {
    if (totalDownWidth) {
      unsigned dataWidth = channel.getType().getDataBitWidth();
      dataType = odsBuilder.getIntegerType(totalDownWidth + dataWidth);
    }
  } else {
    // At most a single downstream extra signal should remain
    if (totalDownWidth != 0) {
      extraSignals.emplace_back(
          MERGED_DOWN_NAME, odsBuilder.getIntegerType(totalDownWidth), true);
    }
  }
  // At most a single upstream extra signal should remain
  if (totalUpWidth != 0) {
    extraSignals.emplace_back(MERGED_UP_NAME,
                              odsBuilder.getIntegerType(totalUpWidth), false);
  }

  odsState.addTypes(ChannelType::get(ctx, dataType, extraSignals));
}

void ReshapeOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                      TypedValue<ChannelType> channel,
                      bool splitDownstreamFromData, Type reshapedType) {
  // Set the operands/attributes
  MLIRContext *ctx = odsBuilder.getContext();
  ChannelReshapeType reshapeType = splitDownstreamFromData
                                       ? ChannelReshapeType::SplitData
                                       : ChannelReshapeType::SplitExtra;
  odsState.addAttribute("reshapeType",
                        ChannelReshapeTypeAttr::get(ctx, reshapeType));
  odsState.addOperands(channel);

  // The compatibility of this type with the input will be checked in the
  // operation's verification function
  odsState.addTypes(reshapedType);
}

LogicalResult ReshapeOp::verify() {
  ChannelType srcType = getChannel().getType(),
              dstType = getReshaped().getType();
  std::pair<unsigned, unsigned> srcExtraWidths = getTotalExtraWidths(srcType),
                                dstExtraWidths = getTotalExtraWidths(dstType);

  // Fails if the merged type has at least one extra downstream signal.
  auto checkNoExtraDownstreamSignal = [&](bool isMerge) -> LogicalResult {
    ChannelType type = isMerge ? dstType : srcType;
    unsigned numSignals = type.getNumDownstreamExtraSignals();
    if (numSignals == 0)
      return success();
    return emitError() << "too many extra downstream signals in the "
                       << (isMerge ? "destination" : "source")
                       << " type, expected 0 but got " << numSignals;
  };

  // Fails if the merged type has more than one extra signal of the specified
  // direction, or if the single extra signal of the direction is incompatible
  // with the split type.
  auto checkMergedExtraSignal = [&](bool isMerge,
                                    bool isDownstream) -> LogicalResult {
    ChannelType type = isMerge ? dstType : srcType;
    unsigned expectedWidth, numExtra;
    StringRef dirStr, combinedName;
    if (isDownstream) {
      expectedWidth = isMerge ? srcExtraWidths.first : dstExtraWidths.first;
      numExtra = type.getNumDownstreamExtraSignals();
      dirStr = "downstream";
      combinedName = MERGED_DOWN_NAME;
    } else {
      expectedWidth = isMerge ? srcExtraWidths.second : dstExtraWidths.second;
      numExtra = type.getNumUpstreamExtraSignals();
      dirStr = "uptream";
      combinedName = MERGED_UP_NAME;
    }

    // There must be either 0 or 1 extra signal of the specified direction
    if (numExtra == 0)
      return success();
    if (numExtra > 1) {
      return emitError() << "merged channel type should have at most one "
                         << dirStr << " signal, but got " << numExtra;
    }

    // Retrieve the single extra signal of the correct direction, then check its
    // name, bitwidth, and type
    const ExtraSignal &extraSignal =
        *llvm::find_if(type.getExtraSignals(), [&](const ExtraSignal &extra) {
          return extra.downstream == isDownstream;
        });

    if (extraSignal.name != combinedName) {
      return emitError() << "invalid name for merged extra " << dirStr
                         << " signal, expected '" << combinedName
                         << "' but got '" << extraSignal.name << "'";
    }
    if (extraSignal.getBitWidth() != expectedWidth) {
      return emitError() << "invalid bitwidth for merged extra " << dirStr
                         << " signal, expected " << expectedWidth << " but got "
                         << extraSignal.getBitWidth();
    }
    if (!isa<IntegerType>(extraSignal.type)) {
      return emitError() << "invalid type for merged extra " << dirStr
                         << " signal, expected IntegerType but got "
                         << extraSignal.type;
    }
    return success();
  };

  // Fails if the merged type's data type is incompatible with the split type.
  auto checkCompatibleDataType = [&](bool isMerge) -> LogicalResult {
    ChannelType splitType, mergeType;
    unsigned extraWidth;
    if (isMerge) {
      splitType = srcType;
      mergeType = dstType;
      extraWidth = srcExtraWidths.first;
    } else {
      splitType = dstType;
      mergeType = srcType;
      extraWidth = dstExtraWidths.first;
    }
    Type mergedDataType = mergeType.getDataType();

    unsigned expectedMergedDataWidth = extraWidth + splitType.getDataBitWidth();
    if (expectedMergedDataWidth != mergeType.getDataBitWidth()) {
      return emitError() << "invalid merged data type bitwidth, expected "
                         << expectedMergedDataWidth << " but got "
                         << mergeType.getDataBitWidth();
    }
    if (extraWidth) {
      if (!isa<IntegerType>(mergedDataType)) {
        return emitError() << "invalid merged data type, expected merged "
                              "IntegerType but got "
                           << mergedDataType;
      }
    } else {
      if (splitType.getDataType() != mergedDataType) {
        return emitError()
               << "invalid destination data type, expected source data type "
               << splitType.getDataType() << " but got " << mergedDataType;
      }
    }
    return success();
  };

  // Fails if the merged type's and split type's data types are different.
  auto checkSameDataType = [&]() -> LogicalResult {
    if (srcType.getDataType() != dstType.getDataType()) {
      return emitError() << "reshaping in this mode should not change "
                            "the data type, expected "
                         << srcType.getDataType() << " but got "
                         << dstType.getDataType();
    }
    return success();
  };

  switch (getReshapeType()) {
  case ChannelReshapeType::MergeData:
    if (failed(checkNoExtraDownstreamSignal(true)) ||
        failed(checkMergedExtraSignal(true, false)) ||
        failed(checkCompatibleDataType(true)))
      return failure();

    break;
  case ChannelReshapeType::MergeExtra:
    if (failed(checkMergedExtraSignal(true, false)) ||
        failed(checkMergedExtraSignal(true, true)) ||
        failed(checkSameDataType()))
      return failure();
    break;
  case ChannelReshapeType::SplitData:
    if (failed(checkNoExtraDownstreamSignal(false)) ||
        failed(checkMergedExtraSignal(false, false)) ||
        failed(checkCompatibleDataType(false)))
      return failure();
    break;
  case ChannelReshapeType::SplitExtra:
    if (failed(checkMergedExtraSignal(false, false)) ||
        failed(checkMergedExtraSignal(false, true)) ||
        failed(checkSameDataType()))
      return failure();
    break;
  }

  return success();
}

OpFoldResult ReshapeOp::fold(FoldAdaptor adaptor) {
  if (getChannel().getType() == getReshaped().getType())
    return getChannel();

  Operation *defOp = getChannel().getDefiningOp();
  if (auto reshapeOp = dyn_cast_if_present<ReshapeOp>(defOp)) {
    if (getReshaped().getType() == reshapeOp.getChannel().getType())
      return reshapeOp.getChannel();
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// CmpFOp
//===----------------------------------------------------------------------===//

LogicalResult
CmpFOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                         ValueRange operands, DictionaryAttr attributes,
                         mlir::OpaqueProperties properties,
                         mlir::RegionRange regions,
                         SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  OpBuilder builder(context);
  inferredReturnTypes.push_back(ChannelType::get(builder.getIntegerType(1)));
  return success();
}

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

LogicalResult
CmpIOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                         ValueRange operands, DictionaryAttr attributes,
                         mlir::OpaqueProperties properties,
                         mlir::RegionRange regions,
                         SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  OpBuilder builder(context);
  inferredReturnTypes.push_back(ChannelType::get(builder.getIntegerType(1)));
  return success();
}

//===----------------------------------------------------------------------===//
// ExtSIOp
//===----------------------------------------------------------------------===//

/// Folds an extension operation if it is preceeded by another extension
/// operation of the same type of if its operand/result types are the same.
template <typename Op>
static OpFoldResult foldExtOp(Op op) {
  if (auto defExtOp = op.getIn().template getDefiningOp<Op>()) {
    // Bypass the preceeding extension operation
    op.getInMutable().assign(defExtOp.getIn());
    return op.getOut();
  }

  unsigned srcWidth = op.getIn().getType().getDataBitWidth();
  unsigned dstWidth = op.getOut().getType().getDataBitWidth();
  if (srcWidth == dstWidth)
    return op.getIn();
  return nullptr;
}

void ExtSIOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<ExtSIOfExtUI>(context);
}

OpFoldResult ExtSIOp::fold(FoldAdaptor adaptor) { return foldExtOp(*this); }

/// Extension operations can only extend to a channel with a wider data type and
/// identical extra signals.
template <typename Op>
static LogicalResult verifyExtOp(Op op) {
  ChannelType srcType = op.getIn().getType();
  ChannelType dstType = op.getOut().getType();

  if (srcType.getDataBitWidth() > dstType.getDataBitWidth()) {
    return op.emitError() << "result channel's data type "
                          << dstType.getDataType()
                          << " must be wider than operand type "
                          << srcType.getDataType();
  }
  return success();
}

LogicalResult ExtSIOp::verify() { return verifyExtOp(*this); }

//===----------------------------------------------------------------------===//
// ExtUIOp
//===----------------------------------------------------------------------===//

OpFoldResult ExtUIOp::fold(FoldAdaptor adaptor) { return foldExtOp(*this); }

LogicalResult ExtUIOp::verify() { return verifyExtOp(*this); }

//===----------------------------------------------------------------------===//
// TruncIOp
//===----------------------------------------------------------------------===//

void TruncIOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<TruncIExtSIToExtSI, TruncIExtUIToExtUI>(context);
}

OpFoldResult TruncIOp::fold(FoldAdaptor adaptor) {
  if (auto defTruncOp = getIn().getDefiningOp<TruncIOp>()) {
    // Bypass the preceeding truncation operation
    getInMutable().assign(defTruncOp.getIn());
    return getResult();
  }

  Operation *defOp = getIn().getDefiningOp();
  if (isa_and_present<ExtSIOp, ExtUIOp>(defOp)) {
    Value src = defOp->getOperand(0);
    unsigned srcWidth = cast<ChannelType>(src.getType()).getDataBitWidth();
    unsigned dstWidth = getOut().getType().getDataBitWidth();
    if (srcWidth > dstWidth) {
      // Bypass the preceeding extension operation
      getInMutable().assign(src);
      return getResult();
    }
    // Bypass the preceeding extension operation and the truncation
    return src;
  }

  // Identical operand and result types mean that the trunc is a no-op
  unsigned srcWidth = getIn().getType().getDataBitWidth();
  unsigned dstWidth = getOut().getType().getDataBitWidth();
  if (srcWidth == dstWidth)
    return getIn();
  return nullptr;
}

LogicalResult TruncIOp::verify() {
  ChannelType srcType = getIn().getType();
  ChannelType dstType = getOut().getType();

  if (srcType.getDataBitWidth() < dstType.getDataBitWidth()) {
    return emitError() << "result channel's data type " << dstType.getDataType()
                       << " must be narrower than operand type "
                       << srcType.getDataType();
  }
  return success();
}

#define GET_OP_CLASSES
#include "dynamatic/Dialect/Handshake/Handshake.cpp.inc"
