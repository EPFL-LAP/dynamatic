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
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
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
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

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

/// Parser for the `custom<SimpleControl>` Tablegen directive.
static ParseResult parseSimpleControl(OpAsmParser &parser, Type &type) {
  // No parsing needed.

  // Make ControlType without any extra signals
  type = ControlType::get(parser.getContext());
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

/// Printer for the `custom<SimpleControl>` Tablegen directive.
static void printSimpleControl(OpAsmPrinter &, Operation *, Type) {
  // No printing needed.
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
                     std::optional<unsigned> numSlots, StringRef bufferType) {
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

  attributes.emplace_back(StringAttr::get(ctx, BUFFER_TYPE_ATTR_NAME),
                          StringAttr::get(ctx, bufferType));

  odsState.addAttribute(RTL_PARAMETERS_ATTR_NAME,
                        DictionaryAttr::get(ctx, attributes));
}

//===----------------------------------------------------------------------===//
// MergeOp
//===----------------------------------------------------------------------===//

OpResult MergeOp::getDataResult() { return cast<OpResult>(getResult()); }

//===----------------------------------------------------------------------===//
// MuxOp
//===----------------------------------------------------------------------===//

bool MuxOp::isControl() {
  return isa<handshake::ControlType>(getResult().getType());
}

ParseResult MuxOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand selectOperand;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> dataOperands;
  handshake::ChannelType selectType;
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();

  // Parse until the type of the select operand
  if (parser.parseOperand(selectOperand) || parser.parseLSquare() ||
      parser.parseOperandList(dataOperands) || parser.parseRSquare() ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseCustomTypeWithFallback(selectType) || parser.parseComma() ||
      parser.parseLSquare())
    return failure();

  int numDataOperands = dataOperands.size();

  // Parse the data operands types
  SmallVector<Type> dataOperandsTypes(numDataOperands);
  for (int i = 0; i < numDataOperands; i++) {
    if (i > 0) {
      if (parser.parseComma())
        return failure();
    }
    if (parseHandshakeType(parser, dataOperandsTypes[i]))
      return failure();
  }

  // Parse the result type
  Type resultType;
  if (parser.parseRSquare() || parser.parseKeyword("to") ||
      parseHandshakeType(parser, resultType))
    return failure();
  result.addTypes(resultType);

  // Fill the result.operands
  return parser.resolveOperands(
      llvm::concat<const OpAsmParser::UnresolvedOperand>(
          ArrayRef<OpAsmParser::UnresolvedOperand>(selectOperand),
          dataOperands),
      llvm::concat<const Type>(ArrayRef<Type>(selectType), dataOperandsTypes),
      allOperandLoc, result.operands);
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
  p << ", [";
  int i = 0;
  for (auto op : getDataOperands()) {
    if (i > 0) {
      p << ", ";
    }
    printHandshakeType(p, op.getType());
    i++;
  }
  p << "] to ";
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
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();

  // Parse until just before the operand types
  if (parser.parseLSquare() || parser.parseOperandList(operands) ||
      parser.parseRSquare() ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseLSquare())
    return failure();

  int numOperands = operands.size();

  // Parse the operand types
  SmallVector<Type> operandTypes(numOperands);
  for (int i = 0; i < numOperands; i++) {
    if (i > 0) {
      if (parser.parseComma())
        return failure();
    }
    if (parseHandshakeType(parser, operandTypes[i]))
      return failure();
  }

  // Parse the return data type
  Type resultDataType;
  if (parser.parseRSquare() || parser.parseKeyword("to") ||
      parseHandshakeType(parser, resultDataType))
    return failure();

  handshake::ChannelType indexType;

  // Parse the index type
  if (parser.parseComma() || parser.parseCustomTypeWithFallback(indexType))
    return failure();

  // Register the result types
  result.addTypes({resultDataType, indexType});

  // Fill the result.operands
  return parser.resolveOperands(operands, operandTypes, allOperandLoc,
                                result.operands);
}

void ControlMergeOp::print(OpAsmPrinter &p) {
  p << " [" << getOperands() << "] ";
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : [";
  int i = 0;
  for (auto op : getOperands()) {
    if (i > 0) {
      p << ", ";
    }
    printHandshakeType(p, op.getType());
    i++;
  }
  p << "] to ";
  printHandshakeType(p, getResult().getType());
  p << ", ";
  p.printStrippedAttrOrType(getIndex().getType());
}

LogicalResult ControlMergeOp::verify() {
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

LogicalResult BufferOp::verify() {
  auto parametersAttr = (*this)->getAttrOfType<DictionaryAttr>("hw.parameters");
  if (!parametersAttr)
    return success();

  auto bufferTypeAttr = parametersAttr.getAs<StringAttr>("BUFFER_TYPE");
  if (!bufferTypeAttr)
    return emitOpError("missing required attribute 'BUFFER_TYPE' in 'hw.parameters'");

  auto numSlotsAttr = parametersAttr.getAs<IntegerAttr>("NUM_SLOTS");
  if (!numSlotsAttr)
    return emitOpError("missing required attribute 'NUM_SLOTS' in 'hw.parameters'");

  StringRef bufferType = bufferTypeAttr.getValue();
  unsigned numSlots = numSlotsAttr.getValue().getZExtValue();

  if ((bufferType == ONE_SLOT_BREAK_DV ||
       bufferType == ONE_SLOT_BREAK_R ||
       bufferType == ONE_SLOT_BREAK_DVR) &&
      numSlots != 1) {
    return emitOpError("buffer type '")
           << bufferType << "' requires NUM_SLOTS = 1, but got " << numSlots;
  }

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

  auto controlType = cast<ControlType>(operands[0].getType());

  // The return type is a ChannelType with:
  // - dataType as specified by the attribute
  // - extra signals matching the input control type
  inferredReturnTypes.push_back(handshake::ChannelType::get(
      attr.getType(), controlType.getExtraSignals()));

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
      } else if (isa<handshake::LoadOp, handshake::StoreOp>(portOp)) {
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

    auto handleLoad = [&](handshake::LoadOp loadOp) -> LogicalResult {
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
          LoadPort(loadOp, input.index(), resIdx++));
      return success();
    };

    auto handleStore = [&](handshake::StoreOp storeOp) -> LogicalResult {
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
      currentGroup->accessPorts.push_back(StorePort(storeOp, input.index()));
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
                .Case<handshake::LoadOp>(handleLoad)
                .Case<handshake::StoreOp>(handleStore)
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

    auto handleLoad = [&](handshake::LoadOp loadOp) -> LogicalResult {
      if (failed(checkAndSetBitwidth(input.value(), lsqPorts.addrWidth)) ||
          failed(checkAndSetBitwidth(memResults[resIdx], lsqPorts.dataWidth)) ||
          failed(checkGroupIsValid()))
        return failure();

      // Add a load port to the group
      currentGroup->accessPorts.push_back(
          LoadPort(loadOp, input.index(), resIdx++));
      --(*currentGroupRemaining);
      return success();
    };

    auto handleStore = [&](handshake::StoreOp storeOp) -> LogicalResult {
      auto dataInput = *(++currentIt);
      if (failed(checkAndSetBitwidth(input.value(), lsqPorts.addrWidth)) ||
          failed(checkAndSetBitwidth(dataInput.value(), lsqPorts.dataWidth)) ||
          failed(checkGroupIsValid()))
        return failure();

      // Add a store port to the group and decrement our group size by one
      currentGroup->accessPorts.push_back(StorePort(storeOp, input.index()));
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
                .Case<handshake::LoadOp>(handleLoad)
                .Case<handshake::StoreOp>(handleStore)
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

SmallVector<Value> LSQOp::getControlPaths(Operation *ctrlOp) {
  // Compute the set of group allocation signals to the LSQ
  DenseSet<Value> lsqGroupAllocs;
  LSQPorts ports = getPorts();
  ValueRange lsqInputs = getOperands();
  for (LSQGroup &group : ports.getGroups())
    lsqGroupAllocs.insert(lsqInputs[group->ctrlPort->getCtrlInputIndex()]);

  // Accumulate all outputs of the control operation that are part of the memory
  // control network; there are typically at most two for any given LSQ
  SmallVector<Value> resultsToAlloc;
  // Control channels to explore, starting from the control operation's results
  SmallVector<Value, 4> controlNet;
  // Control operations already explored from the control operation's results
  // (to avoid looping in the dataflow graph)
  SmallPtrSet<Operation *, 4> controlOps;

  for (OpResult res : ctrlOp->getResults()) {
    // We only care for control-only channels
    if (!isa<handshake::ControlType>(res.getType()))
      continue;

    // Reset the list of control channels to explore and the list of control
    // operations that we have already visited
    controlNet.clear();
    controlOps.clear();

    controlNet.push_back(res);
    controlOps.insert(ctrlOp);
    do {
      Value val = controlNet.pop_back_val();
      auto users = val.getUsers();
      assert(std::distance(users.begin(), users.end()) == 1 &&
             "IR is not materialized");
      Operation *succOp = *users.begin();

      if (lsqGroupAllocs.contains(val)) {
        // We have reached a group allocation to the same LSQ, stop the search
        // along this path
        resultsToAlloc.push_back(res);
        break;
      }

      // Make sure that we do not loop forever over the same control operations
      if (auto [_, newOp] = controlOps.insert(succOp); !newOp)
        continue;

      llvm::TypeSwitch<Operation *, void>(succOp)
          .Case<handshake::ConditionalBranchOp, handshake::BranchOp,
                handshake::MergeOp, handshake::MuxOp, handshake::ForkOp,
                handshake::LazyForkOp, handshake::BufferOp>([&](auto) {
            // If the successor just propagates the control path, add
            // all its results to the list of control channels to
            // explore
            llvm::copy(succOp->getResults(), std::back_inserter(controlNet));
          })
          .Case<handshake::ControlMergeOp>(
              [&](handshake::ControlMergeOp cmergeOp) {
                // Only the control merge's data output forwards the input
                controlNet.push_back(cmergeOp.getResult());
              });
    } while (!controlNet.empty());
  }

  return resultsToAlloc;
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

handshake::MemoryOpInterface dynamatic::findMemInterface(Value val) {
  llvm::SmallDenseSet<Operation *, 2> explore(val.getUsers().begin(),
                                              val.getUsers().end());
  llvm::SmallDenseSet<Operation *, 2> visited;
  for (Operation *op : explore) {
    if (auto [_, newOp] = visited.insert(op); !newOp)
      continue;
    if (auto memOp = dyn_cast<handshake::MemoryOpInterface>(op))
      return memOp;
    if (isa<handshake::ExtSIOp, handshake::ExtUIOp, handshake::TruncIOp,
            handshake::ForkOp, handshake::LazyForkOp>(op))
      explore.insert(op->getUsers().begin(), op->getUsers().end());
  }
  return nullptr;
}

MemoryPort::MemoryPort(Operation *portOp, ArrayRef<unsigned> oprdIndices,
                       ArrayRef<unsigned> resIndices, Kind kind)
    : portOp(portOp), oprdIndices(oprdIndices), resIndices(resIndices),
      kind(kind) {}

ControlPort::ControlPort(Operation *ctrlOp, unsigned ctrlInputIdx)
    : MemoryPort(ctrlOp, {ctrlInputIdx}, {}, Kind::CONTROL) {}

LoadPort::LoadPort(handshake::LoadOp loadOp, unsigned addrInputIdx,
                   unsigned dataOutputIdx)
    : MemoryPort(loadOp, {addrInputIdx}, {dataOutputIdx}, Kind::LOAD) {}

handshake::LoadOp LoadPort::getLoadOp() const {
  return cast<handshake::LoadOp>(portOp);
}

StorePort::StorePort(handshake::StoreOp storeOp, unsigned addrInputIdx)
    : MemoryPort(storeOp, {addrInputIdx, addrInputIdx + 1}, {}, Kind::STORE){};

handshake::StoreOp StorePort::getStoreOp() const {
  return cast<handshake::StoreOp>(portOp);
}

LSQLoadStorePort::LSQLoadStorePort(dynamatic::handshake::LSQOp lsqOp,
                                   unsigned loadAddrInputIdx,
                                   unsigned loadDataOutputIdx)
    : MemoryPort(lsqOp,
                 {loadAddrInputIdx, loadAddrInputIdx + 1, loadAddrInputIdx + 2},
                 {loadDataOutputIdx}, Kind::LSQ_LOAD_STORE) {}

handshake::LSQOp LSQLoadStorePort::getLSQOp() const {
  return cast<handshake::LSQOp>(portOp);
}

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

GroupMemoryPorts::GroupMemoryPorts(ControlPort ctrlPort) : ctrlPort(ctrlPort){};

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

namespace {
using FGetIndices = std::function<ArrayRef<unsigned>(const MemoryPort &)>;
} // namespace

static ValueRange getInterfaceValueRange(const FuncMemoryPorts &ports,
                                         ValueRange allValues,
                                         const FGetIndices &fGetIndices) {
  size_t firstIdx = std::string::npos;
  for (const MemoryPort &port : ports.interfacePorts) {
    if (auto indices = fGetIndices(port); !indices.empty()) {
      firstIdx = indices.front();
      break;
    }
  }
  if (firstIdx == std::string::npos)
    return {};

  for (const MemoryPort &port : llvm::reverse(ports.interfacePorts)) {
    if (auto indices = fGetIndices(port); !indices.empty())
      return allValues.slice(firstIdx, indices.back() - firstIdx + 1);
  }
  llvm_unreachable("no last index, this is impossible");
  return {};
}

ValueRange FuncMemoryPorts::getInterfacesInputs() {
  return getInterfaceValueRange(*this, memOp->getOperands(),
                                &MemoryPort::getOprdIndices);
}

ValueRange FuncMemoryPorts::getInterfacesResults() {
  return getInterfaceValueRange(*this, memOp->getResults(),
                                &MemoryPort::getResIndices);
}

MCBlock::MCBlock(GroupMemoryPorts *group, unsigned blockID)
    : blockID(blockID), group(group){};

MCPorts::MCPorts(handshake::MemoryControllerOp mcOp) : FuncMemoryPorts(mcOp){};

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
  assert(connectsToLSQ() && "no LSQ connected");
  std::optional<LSQLoadStorePort> lsqPort =
      dyn_cast<LSQLoadStorePort>(interfacePorts.front());
  assert(lsqPort && "lsq load/store port undefined");
  return *lsqPort;
}

LSQGroup::LSQGroup(GroupMemoryPorts *group, unsigned groupID)
    : groupID(groupID), group(group) {}

SmallVector<LSQGroup> LSQPorts::getGroups() {
  SmallVector<LSQGroup> lsqGroups;
  for (auto [idx, ports] : llvm::enumerate(groups))
    lsqGroups.emplace_back(&ports, idx);
  return lsqGroups;
}

LSQPorts::LSQPorts(handshake::LSQOp lsqOp) : FuncMemoryPorts(lsqOp){};

handshake::LSQOp LSQPorts::getLSQOp() const {
  return cast<handshake::LSQOp>(memOp);
}

MCLoadStorePort LSQPorts::getMCPort() const {
  assert(connectsToMC() && "no MC connected");
  std::optional<MCLoadStorePort> mcPort =
      dyn_cast<MCLoadStorePort>(interfacePorts.front());
  assert(mcPort && "mc load/store port undefined");
  return *mcPort;
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
  assert(cast<handshake::ControlType>(ctrl.getType()).getNumExtraSignals() ==
             0 &&
         "expected ctrl to have no extra signals");
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
  p << " : ";
  printHandshakeType(p, getChannelLike().getType());
  p << " to _ ";
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
// CmpFOp
//===----------------------------------------------------------------------===//

LogicalResult
CmpFOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                         ValueRange operands, DictionaryAttr attributes,
                         mlir::OpaqueProperties properties,
                         mlir::RegionRange regions,
                         SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  OpBuilder builder(context);
  inferredReturnTypes.push_back(ChannelType::get(
      builder.getIntegerType(1),
      // We can assume that
      // - operand[0] exists
      // - operand[0] is a channel type
      // - all operands have the same extra signals
      // Note that this cast throws an error if the assumption is not met
      operands[0].getType().cast<ChannelType>().getExtraSignals()));
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
  inferredReturnTypes.push_back(ChannelType::get(
      builder.getIntegerType(1),
      // We can assume that
      // - operand[0] exists
      // - operand[0] is a channel type
      // - all operands have the same extra signals
      // Note that this cast throws an error if the assumption is not met
      operands[0].getType().cast<ChannelType>().getExtraSignals()));
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
/// Extension operations can only extend to a channel with a wider data type and
/// identical extra signals.
template <typename Op>
static LogicalResult verifyTruncOp(Op op) {
  ChannelType srcType = op.getIn().getType();
  ChannelType dstType = op.getOut().getType();

  if (srcType.getDataBitWidth() < dstType.getDataBitWidth()) {
    return op.emitError() << "result channel's data type "
                          << dstType.getDataType()
                          << " must be narrower than operand type "
                          << srcType.getDataType();
  }
  return success();
}

LogicalResult TruncIOp::verify() { return verifyTruncOp(*this); }

//===----------------------------------------------------------------------===//
// TruncFOp
//===----------------------------------------------------------------------===//

LogicalResult TruncFOp::verify() { return verifyTruncOp(*this); }

//===----------------------------------------------------------------------===//
// ExtFOp
//===----------------------------------------------------------------------===//
LogicalResult ExtFOp::verify() { return verifyExtOp(*this); }

#define GET_OP_CLASSES
#include "dynamatic/Dialect/Handshake/Handshake.cpp.inc"