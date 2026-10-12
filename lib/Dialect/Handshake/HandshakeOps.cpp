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

static IntegerAttr constantFoldExt(Operation *op, Attribute attr) {
  auto integerAttr = cast<IntegerAttr>(attr);
  if (auto extUI = dyn_cast<ExtUIOp>(op))
    return IntegerAttr::get(
        extUI.getType().getDataType(),
        integerAttr.getValue().zext(extUI.getType().getDataBitWidth()));

  if (auto extSI = dyn_cast<ExtSIOp>(op))
    return IntegerAttr::get(
        extSI.getType().getDataType(),
        integerAttr.getValue().sext(extSI.getType().getDataBitWidth()));

  llvm_unreachable("only expected extui and extsi");
}

static IntegerAttr constantFoldTruncI(Type resultType, Attribute attr) {
  unsigned bitwidth = cast<ChannelType>(resultType).getDataBitWidth();
  return IntegerAttr::get(IntegerType::get(resultType.getContext(), bitwidth),
                          cast<IntegerAttr>(attr).getValue().trunc(bitwidth));
}

namespace {
#include "lib/Dialect/Handshake/HandshakeCanonicalization.inc"
} // namespace

void handshake::HandshakeDialect::getCanonicalizationPatterns(
    RewritePatternSet &set) const {
  populateWithGenerated(set);
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
  // this is additional verification
  // so both attributes have already been verified as present
  int numSlots = getNumSlots();
  int64_t dvLatency = getLatencyDV();
  BufferType bufferType = getBufferType();

  if ((bufferType == BufferType::ONE_SLOT_BREAK_DV ||
       bufferType == BufferType::ONE_SLOT_BREAK_R ||
       bufferType == BufferType::ONE_SLOT_BREAK_DVR) &&
      numSlots != 1) {
    return emitOpError("buffer type '")
           << stringifyEnum(bufferType) << "' requires NUM_SLOTS = 1, but got "
           << numSlots;
  }

  auto emitLatencyError = [&](int64_t expectedLatency) -> LogicalResult {
    return emitOpError("buffer type '")
           << stringifyEnum(bufferType)
           << "' requires DV_LATENCY = " << expectedLatency << ", but got "
           << dvLatency;
  };

  switch (bufferType) {
  case BufferType::ONE_SLOT_BREAK_R:
  case BufferType::FIFO_BREAK_NONE:
    if (dvLatency != 0)
      return emitLatencyError(0);
    break;
  case BufferType::ONE_SLOT_BREAK_DV:
  case BufferType::FIFO_BREAK_DV:
  case BufferType::ONE_SLOT_BREAK_DVR:
    if (dvLatency != 1)
      return emitLatencyError(1);
    break;
  case BufferType::SHIFT_REG_BREAK_DV:
    if (dvLatency != numSlots) {
      return emitOpError("buffer type '")
             << stringifyEnum(bufferType)
             << "' requires DV_LATENCY = NUM_SLOTS (" << numSlots
             << "), but got " << dvLatency;
    }
    break;
  case BufferType::COUNTER_BUFFER:
    if (numSlots != 1) {
      return emitOpError("buffer type '")
             << stringifyEnum(bufferType)
             << "' stores a single token and requires NUM_SLOTS = 1, but got "
             << numSlots;
    }
    if (dvLatency < 1) {
      return emitOpError("buffer type '")
             << stringifyEnum(bufferType)
             << "' requires DV_LATENCY >= 1, but got " << dvLatency;
    }
    break;
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

/// Finds the operation that produces the input by backtracking the def-use
/// chain through potential bitwidth modification and/or fork operations.
/// Returns nullptr if the input is a block argument.
static Operation *backtrackToMemInput(Value input) {
  Operation *inputOp = input.getDefiningOp();
  while (isa_and_present<handshake::ExtSIOp, handshake::ExtUIOp,
                         handshake::TruncIOp, handshake::ForkOp>(inputOp))
    inputOp = inputOp->getOperand(0).getDefiningOp();
  return inputOp;
}

static handshake::ChannelType wrapChannel(Type type) {
  assert(ChannelType::isSupportedSignalType(type) && "unsupported data type");
  return handshake::ChannelType::get(type);
}

/// Name of the attribute holding the groups of memory interfaces.
static constexpr StringLiteral GROUPS_ATTR_NAME("groups");

/// Converts a list of memory groups into an array attribute.
static ArrayAttr buildGroupsAttr(OpBuilder &builder,
                                 ArrayRef<MemGroupAttr> groups) {
  return builder.getArrayAttr(
      SmallVector<Attribute>(groups.begin(), groups.end()));
}

/// Name of the attributes holding the operand/result segment sizes of memory
/// interfaces.
static StringRef getOprdSegmentsAttrName() {
  return OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr();
}
static StringRef getResSegmentsAttrName() {
  return OpTrait::AttrSizedResultSegments<void>::getResultSegmentSizeAttr();
}

/// Returns the operand numbers of the operands in the range.
static SmallVector<unsigned> getOperandNumbers(MutableOperandRange operands) {
  return llvm::to_vector(llvm::map_range(
      operands, [](OpOperand &oprd) { return oprd.getOperandNumber(); }));
}

/// Returns the result numbers of the results in the range.
static SmallVector<unsigned> getResultNumbers(ValueRange results) {
  return llvm::to_vector(llvm::map_range(results, [](Value res) {
    return cast<OpResult>(res).getResultNumber();
  }));
}

/// Adds the control and access ports of all groups to the memory interface's
/// ports. The i-th control (resp. load, store) port of the groups corresponds
/// to the i-th operand(s) of the control (resp. load, store) segment(s).
static void addGroupPorts(FuncMemoryPorts &ports, ArrayRef<MemGroupAttr> groups,
                          MutableOperandRange ctrls,
                          MutableOperandRange ldAddrs,
                          MutableOperandRange stAddrs,
                          MutableOperandRange stDatas, ValueRange ldDatas) {
  Operation *memOp = ports.memOp;
  SmallVector<unsigned> ctrlIdxs = getOperandNumbers(ctrls);
  SmallVector<unsigned> ldAddrIdxs = getOperandNumbers(ldAddrs);
  SmallVector<unsigned> stAddrIdxs = getOperandNumbers(stAddrs);
  SmallVector<unsigned> stDataIdxs = getOperandNumbers(stDatas);
  SmallVector<unsigned> ldDataIdxs = getResultNumbers(ldDatas);

  unsigned ctrlIdx = 0, ldIdx = 0, stIdx = 0;
  for (MemGroupAttr group : groups) {
    if (group.getHasCtrl()) {
      unsigned idx = ctrlIdxs[ctrlIdx++];
      Operation *ctrlOp = backtrackToMemInput(memOp->getOperand(idx));
      ports.groups.emplace_back(ControlPort(ctrlOp, idx));
    } else {
      ports.groups.emplace_back();
    }
    GroupMemoryPorts &groupPorts = ports.groups.back();
    for (MemAccessKind kind : group.getAccesses()) {
      if (kind == MemAccessKind::LOAD) {
        unsigned addrIdx = ldAddrIdxs[ldIdx];
        auto loadOp = dyn_cast_if_present<handshake::LoadOp>(
            backtrackToMemInput(memOp->getOperand(addrIdx)));
        groupPorts.accessPorts.push_back(
            LoadPort(loadOp, addrIdx, ldDataIdxs[ldIdx++]));
      } else {
        unsigned addrIdx = stAddrIdxs[stIdx];
        auto storeOp = dyn_cast_if_present<handshake::StoreOp>(
            backtrackToMemInput(memOp->getOperand(addrIdx)));
        groupPorts.accessPorts.push_back(
            StorePort(storeOp, addrIdx, stDataIdxs[stIdx++]));
      }
    }
  }
}

/// Verifies that the groups of a memory interface are consistent with the
/// sizes of its control, load, and store segments.
static LogicalResult verifyMemGroups(Operation *memOp,
                                     ArrayRef<MemGroupAttr> groups,
                                     unsigned numCtrls, unsigned numLdAddrs,
                                     unsigned numLdDatas, unsigned numStAddrs,
                                     unsigned numStDatas) {
  if (groups.empty())
    return memOp->emitError() << "memory interface must have at least one "
                                 "group";

  unsigned groupCtrls = 0, groupLoads = 0, groupStores = 0;
  for (MemGroupAttr group : groups) {
    if (group.getHasCtrl())
      ++groupCtrls;
    groupLoads += group.getNumLoads();
    groupStores += group.getNumStores();
  }
  auto checkCount = [&](StringRef what, unsigned expected,
                        unsigned actual) -> LogicalResult {
    if (expected == actual)
      return success();
    return memOp->emitError() << "groups declare " << expected << " " << what
                              << ", but " << actual << " are provided";
  };
  return success(
      succeeded(checkCount("control ports", groupCtrls, numCtrls)) &&
      succeeded(checkCount("load addresses", groupLoads, numLdAddrs)) &&
      succeeded(checkCount("load data results", groupLoads, numLdDatas)) &&
      succeeded(checkCount("store addresses", groupStores, numStAddrs)) &&
      succeeded(checkCount("store data", groupStores, numStDatas)));
}

/// Verifies that all non-null values have the same bitwidth.
static LogicalResult verifySameBitwidth(Operation *memOp, StringRef what,
                                        ArrayRef<Value> values) {
  std::optional<unsigned> width;
  for (Value val : values) {
    if (!val)
      continue;
    unsigned valWidth = getHandshakeTypeBitWidth(val.getType());
    if (!width) {
      width = valWidth;
    } else if (*width != valWidth) {
      return memOp->emitError()
             << "inconsistent bitwidths for " << what << ", expected " << *width
             << " bits but got " << valWidth << " bits";
    }
  }
  return success();
}

/// Returns the bitwidth of the first non-null value, or 0 if there is none.
static unsigned getFirstBitwidth(ArrayRef<Value> values) {
  for (Value val : values) {
    if (val)
      return getHandshakeTypeBitWidth(val.getType());
  }
  return 0;
}

/// Prints a comma-separated list of operands in parentheses after the keyword,
/// unless the list is empty.
static void printMemSection(OpAsmPrinter &p, StringRef keyword,
                            ValueRange values) {
  if (values.empty())
    return;
  p << " " << keyword << "(";
  llvm::interleaveComma(values, p);
  p << ")";
}

/// Parses an optional `keyword(operand, ...)` section, appending the operands
/// to the list. Returns the number of parsed operands.
static FailureOr<unsigned>
parseMemSection(OpAsmParser &parser, StringRef keyword,
                SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands) {
  if (failed(parser.parseOptionalKeyword(keyword)))
    return 0;
  size_t numBefore = operands.size();
  if (parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren))
    return failure();
  return operands.size() - numBefore;
}

/// Parses an optional reference to a memory region of the form
/// `[%memref : type]`. If the `allowMC` flag is set, `[MC]` is accepted instead
/// to indicate the absence of a memory region (sets `hasMemRef` to false).
static ParseResult
parseMemRegion(OpAsmParser &parser, bool allowMC, bool &hasMemRef,
               SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
               SmallVectorImpl<Type> &types) {
  if (parser.parseLSquare())
    return failure();
  if (allowMC && succeeded(parser.parseOptionalKeyword("MC"))) {
    hasMemRef = false;
    return parser.parseRSquare();
  }
  hasMemRef = true;
  OpAsmParser::UnresolvedOperand memref;
  Type memrefType;
  if (parser.parseOperand(memref) || parser.parseColonType(memrefType) ||
      parser.parseRSquare())
    return failure();
  operands.push_back(memref);
  types.push_back(memrefType);
  return success();
}

/// Parses the attribute dictionary and the functional type of a memory
/// interface, then resolves the operands (the memory region's operand and
/// type, if any, must already be at the front of the lists).
static ParseResult
parseMemTypes(OpAsmParser &parser, OperationState &result, llvm::SMLoc loc,
              SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
              SmallVectorImpl<Type> &operandTypes) {
  FunctionType funType;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(funType))
    return failure();
  llvm::append_range(operandTypes, funType.getInputs());
  result.addTypes(funType.getResults());
  return parser.resolveOperands(operands, operandTypes, loc, result.operands);
}

/// Sets the result segment sizes of a parsed memory interface, checking that
/// they are consistent with the number of parsed result types.
static ParseResult setParsedResultSegments(OpAsmParser &parser,
                                           OperationState &result,
                                           llvm::SMLoc loc,
                                           ArrayRef<int32_t> segments) {
  int64_t numResults = 0;
  for (int32_t size : segments)
    numResults += size;
  if (numResults != static_cast<int64_t>(result.types.size()))
    return parser.emitError(loc)
           << "expected " << numResults
           << " results based on the memory groups and operands, but got "
           << result.types.size();
  result.addAttribute(getResSegmentsAttrName(),
                      parser.getBuilder().getDenseI32ArrayAttr(segments));
  return success();
}

/// Returns the memory groups stored in a parsed attribute dictionary.
static FailureOr<SmallVector<MemGroupAttr>>
getParsedGroups(OpAsmParser &parser, OperationState &result, llvm::SMLoc loc) {
  // Both memory interfaces store their groups under the same attribute name
  auto groupsAttr =
      dyn_cast_if_present<ArrayAttr>(result.attributes.get(GROUPS_ATTR_NAME));
  if (!groupsAttr)
    return parser.emitError(loc) << "expected 'groups' attribute";
  SmallVector<MemGroupAttr> groups;
  for (Attribute attr : groupsAttr) {
    auto group = dyn_cast<MemGroupAttr>(attr);
    if (!group)
      return parser.emitError(loc)
             << "expected 'groups' to only contain memory groups";
    groups.push_back(group);
  }
  return groups;
}

//===----------------------------------------------------------------------===//
// MemoryControllerOp
//===----------------------------------------------------------------------===//

void MemoryControllerOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                               Value memRef, Value memStart, ValueRange ctrls,
                               ValueRange ldAddrs, ValueRange stAddrs,
                               ValueRange stDatas, Value ctrlEnd,
                               ArrayRef<MemGroupAttr> groups) {
  build(odsBuilder, odsState, memRef, memStart, ctrls, ldAddrs, stAddrs,
        stDatas, nullptr, nullptr, nullptr, ctrlEnd, groups);
}

void MemoryControllerOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                               Value memRef, Value memStart, ValueRange ctrls,
                               ValueRange ldAddrs, ValueRange stAddrs,
                               ValueRange stDatas, Value lsqLdAddr,
                               Value lsqStAddr, Value lsqStData, Value ctrlEnd,
                               ArrayRef<MemGroupAttr> groups) {
  bool hasLSQ = lsqLdAddr != nullptr;
  assert((hasLSQ == (lsqStAddr != nullptr)) &&
         (hasLSQ == (lsqStData != nullptr)) &&
         "LSQ ports must all be present or absent");

  // Operands
  odsState.addOperands({memRef, memStart});
  odsState.addOperands(ctrls);
  odsState.addOperands(ldAddrs);
  odsState.addOperands(stAddrs);
  odsState.addOperands(stDatas);
  if (hasLSQ)
    odsState.addOperands({lsqLdAddr, lsqStAddr, lsqStData});
  odsState.addOperands(ctrlEnd);
  int32_t lsqSize = hasLSQ ? 1 : 0;
  odsState.addAttribute(
      getOprdSegmentsAttrName(),
      odsBuilder.getDenseI32ArrayAttr({1, 1, static_cast<int32_t>(ctrls.size()),
                                       static_cast<int32_t>(ldAddrs.size()),
                                       static_cast<int32_t>(stAddrs.size()),
                                       static_cast<int32_t>(stDatas.size()),
                                       lsqSize, lsqSize, lsqSize, 1}));

  // Results (data outputs get their type from the memref)
  MemRefType memrefType = memRef.getType().cast<MemRefType>();
  Type dataType = wrapChannel(memrefType.getElementType());
  odsState.types.append(ldAddrs.size(), dataType);
  if (hasLSQ)
    odsState.types.push_back(dataType);
  odsState.types.push_back(
      handshake::ControlType::get(odsBuilder.getContext()));
  odsState.addAttribute(
      getResSegmentsAttrName(),
      odsBuilder.getDenseI32ArrayAttr(
          {static_cast<int32_t>(ldAddrs.size()), lsqSize, 1}));

  odsState.addAttribute(GROUPS_ATTR_NAME, buildGroupsAttr(odsBuilder, groups));
}

ParseResult MemoryControllerOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<Type> operandTypes;
  llvm::SMLoc loc = parser.getCurrentLocation();

  bool hasMemRef;
  if (parseMemRegion(parser, /*allowMC=*/false, hasMemRef, operands,
                     operandTypes))
    return failure();

  // Parse operand sections, in segment order
  SmallVector<int32_t> segments{1};
  bool hasLSQ = false;
  for (StringRef keyword :
       {"start", "ctrl", "ldAddr", "stAddr", "stData", "fromLSQ", "end"}) {
    FailureOr<unsigned> numOprds = parseMemSection(parser, keyword, operands);
    if (failed(numOprds))
      return failure();
    if (keyword == "fromLSQ") {
      if (*numOprds != 0 && *numOprds != 3)
        return parser.emitError(loc)
               << "expected 3 operands (load address, store address, store "
                  "data) from LSQ";
      hasLSQ = *numOprds != 0;
      segments.append(3, hasLSQ ? 1 : 0);
    } else {
      segments.push_back(*numOprds);
    }
  }
  result.addAttribute(getOprdSegmentsAttrName(),
                      parser.getBuilder().getDenseI32ArrayAttr(segments));

  if (parseMemTypes(parser, result, loc, operands, operandTypes))
    return failure();

  // Derive the result segments from the groups
  FailureOr<SmallVector<MemGroupAttr>> groups =
      getParsedGroups(parser, result, loc);
  if (failed(groups))
    return failure();
  int32_t numLoads = 0;
  for (MemGroupAttr group : *groups)
    numLoads += group.getNumLoads();
  return setParsedResultSegments(parser, result, loc,
                                 {numLoads, hasLSQ ? 1 : 0, 1});
}

void MemoryControllerOp::print(OpAsmPrinter &p) {
  p << "[" << getMemRef() << " : " << getMemRef().getType() << "]";
  printMemSection(p, "start", getMemStart());
  printMemSection(p, "ctrl", getCtrls());
  printMemSection(p, "ldAddr", getLdAddrs());
  printMemSection(p, "stAddr", getStAddrs());
  printMemSection(p, "stData", getStDatas());
  if (isConnectedToLSQ()) {
    printMemSection(p, "fromLSQ",
                    ValueRange{getLsqLdAddr(), getLsqStAddr(), getLsqStData()});
  }
  printMemSection(p, "end", getCtrlEnd());
  p.printOptionalAttrDict((*this)->getAttrs(), {getOprdSegmentsAttrName(),
                                                getResSegmentsAttrName()});
  p << " : ";
  p.printFunctionalType(getOperands().drop_front().getTypes(),
                        getResultTypes());
}

/// Derives the ports of a memory controller from its operand segments and
/// groups.
static void getMCPorts(MCPorts &mcPorts) {
  handshake::MemoryControllerOp mcOp = mcPorts.getMCOp();
  addGroupPorts(mcPorts, mcOp.getMemGroups(), mcOp.getCtrlsMutable(),
                mcOp.getLdAddrsMutable(), mcOp.getStAddrsMutable(),
                mcOp.getStDatasMutable(), mcOp.getLdDatas());

  if (mcOp.isConnectedToLSQ()) {
    auto lsqOp = dyn_cast_if_present<handshake::LSQOp>(
        backtrackToMemInput(mcOp.getLsqLdAddr()));
    mcPorts.interfacePorts.push_back(LSQLoadStorePort(
        lsqOp, mcOp.getLsqLdAddrMutable()[0].getOperandNumber(),
        cast<OpResult>(mcOp.getLsqLdData()).getResultNumber()));
  }

  SmallVector<Value> ctrls(mcOp.getCtrls());
  SmallVector<Value> addrs(mcOp.getLdAddrs());
  llvm::append_range(addrs, mcOp.getStAddrs());
  addrs.append({mcOp.getLsqLdAddr(), mcOp.getLsqStAddr()});
  SmallVector<Value> datas(mcOp.getStDatas());
  llvm::append_range(datas, mcOp.getLdDatas());
  datas.append({mcOp.getLsqStData(), mcOp.getLsqLdData()});
  mcPorts.ctrlWidth = getFirstBitwidth(ctrls);
  mcPorts.addrWidth = getFirstBitwidth(addrs);
  mcPorts.dataWidth = getFirstBitwidth(datas);
}

LogicalResult MemoryControllerOp::verify() {
  SmallVector<MemGroupAttr> groups = getMemGroups();
  if (failed(verifyMemGroups(*this, groups, getCtrls().size(),
                             getLdAddrs().size(), getLdDatas().size(),
                             getStAddrs().size(), getStDatas().size())))
    return failure();

  // Blocks must be unique, and blocks with stores must have a control port
  DenseSet<unsigned> uniqueBlocks;
  for (MemGroupAttr group : groups) {
    if (auto [_, newBlock] = uniqueBlocks.insert(group.getId()); !newBlock)
      return emitError() << "Block IDs must be unique but ID " << group.getId()
                         << " was found at least twice in the list.";
    if (group.getNumStores() != 0 && !group.getHasCtrl())
      return emitError() << "Block " << group.getId()
                         << " has stores but no control port.";
  }

  // Access ports must belong to the basic block of their group
  MCPorts mcPorts = getPorts();
  for (auto [group, groupPorts] : llvm::zip_equal(groups, mcPorts.groups)) {
    for (MemoryPort &port : groupPorts.accessPorts) {
      if (!port.portOp)
        continue;
      // Read the attribute directly since the dialect does not link against
      // the support library (where `getLogicBB` is defined)
      auto bb = port.portOp->getAttrOfType<IntegerAttr>(BB_ATTR_NAME);
      if (bb && bb.getUInt() != group.getId())
        return emitError() << "Access port belongs to block " << bb.getUInt()
                           << ", but expected it to be part of block "
                           << group.getId();
    }
  }

  // The ports from/to the LSQ must all be present or absent
  bool hasLSQ = isConnectedToLSQ();
  if (hasLSQ != (getLsqStAddr() != nullptr) ||
      hasLSQ != (getLsqStData() != nullptr) ||
      hasLSQ != (getLsqLdData() != nullptr))
    return emitError() << "LSQ load address, store address, store data, and "
                          "load data ports must all be present or absent.";
  if (hasLSQ) {
    for (Value lsqVal :
         {getLsqLdAddr(), getLsqStAddr(), getLsqStData(), getLsqLdData()}) {
      if (cast<ExtraSignalsTypeInterface>(lsqVal.getType())
              .getNumExtraSignals() != 0)
        return emitError() << "LSQ ports shouldn't have any extra signals.";
    }
  }

  // Signals of the same nature must have the same bitwidth
  SmallVector<Value> addrs(getLdAddrs());
  llvm::append_range(addrs, getStAddrs());
  addrs.append({getLsqLdAddr(), getLsqStAddr()});
  SmallVector<Value> datas(getStDatas());
  llvm::append_range(datas, getLdDatas());
  datas.append({getLsqStData(), getLsqLdData()});
  return success(succeeded(verifySameBitwidth(
                     *this, "control ports", SmallVector<Value>(getCtrls()))) &&
                 succeeded(verifySameBitwidth(*this, "addresses", addrs)) &&
                 succeeded(verifySameBitwidth(*this, "data", datas)));
}

dynamatic::MCPorts MemoryControllerOp::getPorts() {
  MCPorts mcPorts(*this);
  getMCPorts(mcPorts);
  return mcPorts;
}

size_t MemoryControllerOp::getNumLoadPorts() {
  MCPorts mcPorts = getPorts();
  const FuncMemoryPorts &ports = mcPorts;
  size_t numLoads = 0;
  for (const GroupMemoryPorts &blockPorts : ports.groups) {
    for (const MemoryPort &accessPort : blockPorts.accessPorts) {
      if (std::optional<LoadPort> loadPort = dyn_cast<LoadPort>(accessPort)) {
        ++numLoads;
      }
    }
  }
  return numLoads;
}

std::optional<LoadPort> MemoryControllerOp::getLoadPort(size_t index) {
  MCPorts mcPorts = getPorts();
  const FuncMemoryPorts &ports = mcPorts;
  size_t loadIdx = 0;
  for (const GroupMemoryPorts &blockPorts : ports.groups) {
    for (const MemoryPort &accessPort : blockPorts.accessPorts) {
      if (std::optional<LoadPort> loadPort = dyn_cast<LoadPort>(accessPort)) {
        if (loadIdx == index) {
          return loadPort;
        }
        ++loadIdx;
      }
    }
  }
  return std::nullopt;
}

MemoryControllerSlotNamer
MemoryControllerOp::getLoadPortSlotNamer(size_t index) {
  size_t nLoads = getNumLoadPorts();
  assert(index < nLoads);
  auto name = getOperation()->getAttrOfType<::mlir::StringAttr>(
      NameAnalysis::ATTR_NAME);
  assert(name && "name required for memory controller slot namer");
  MemoryControllerSlotNamer ret(MemoryControllerSlotNamer::PortType::Load,
                                name.str(), index);
  return ret;
}

//===----------------------------------------------------------------------===//
// LSQOp
//===----------------------------------------------------------------------===//

void LSQOp::build(OpBuilder &odsBuilder, OperationState &odsState, Value memRef,
                  Value memStart, ValueRange ctrls, ValueRange ldAddrs,
                  ValueRange stAddrs, ValueRange stDatas, Value ctrlEnd,
                  ArrayRef<MemGroupAttr> groups) {
  // Operands
  odsState.addOperands({memRef, memStart});
  odsState.addOperands(ctrls);
  odsState.addOperands(ldAddrs);
  odsState.addOperands(stAddrs);
  odsState.addOperands(stDatas);
  odsState.addOperands(ctrlEnd);
  odsState.addAttribute(getOprdSegmentsAttrName(),
                        odsBuilder.getDenseI32ArrayAttr(
                            {1, 1, static_cast<int32_t>(ctrls.size()),
                             static_cast<int32_t>(ldAddrs.size()),
                             static_cast<int32_t>(stAddrs.size()),
                             static_cast<int32_t>(stDatas.size()), 0, 1}));

  // Results (data outputs get their type from the memref)
  MemRefType memrefType = memRef.getType().cast<MemRefType>();
  odsState.types.append(ldAddrs.size(),
                        wrapChannel(memrefType.getElementType()));
  odsState.types.push_back(
      handshake::ControlType::get(odsBuilder.getContext()));
  odsState.addAttribute(
      getResSegmentsAttrName(),
      odsBuilder.getDenseI32ArrayAttr(
          {static_cast<int32_t>(ldAddrs.size()), 0, 0, 0, 1}));

  odsState.addAttribute(GROUPS_ATTR_NAME, buildGroupsAttr(odsBuilder, groups));
}

void LSQOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                  handshake::MemoryControllerOp mcOp, ValueRange ctrls,
                  ValueRange ldAddrs, ValueRange stAddrs, ValueRange stDatas,
                  Value ldDataFromMC, ArrayRef<MemGroupAttr> groups) {
  // Operands
  odsState.addOperands(ctrls);
  odsState.addOperands(ldAddrs);
  odsState.addOperands(stAddrs);
  odsState.addOperands(stDatas);
  odsState.addOperands(ldDataFromMC);
  odsState.addAttribute(getOprdSegmentsAttrName(),
                        odsBuilder.getDenseI32ArrayAttr(
                            {0, 0, static_cast<int32_t>(ctrls.size()),
                             static_cast<int32_t>(ldAddrs.size()),
                             static_cast<int32_t>(stAddrs.size()),
                             static_cast<int32_t>(stDatas.size()), 1, 0}));

  // Results (data outputs get their type from the memref). The LSQ is a slave
  // interface in this case (the MC is the master), so it doesn't produce a
  // completion signal but instead load/store requests going to the MC
  MLIRContext *ctx = odsBuilder.getContext();
  Type dataType = wrapChannel(mcOp.getMemRefType().getElementType());
  Type addrType = handshake::ChannelType::getAddrChannel(ctx);
  odsState.types.append(ldAddrs.size(), dataType);
  odsState.types.append({addrType, addrType, dataType});
  odsState.addAttribute(
      getResSegmentsAttrName(),
      odsBuilder.getDenseI32ArrayAttr(
          {static_cast<int32_t>(ldAddrs.size()), 1, 1, 1, 0}));

  odsState.addAttribute(GROUPS_ATTR_NAME, buildGroupsAttr(odsBuilder, groups));
}

ParseResult LSQOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<Type> operandTypes;
  llvm::SMLoc loc = parser.getCurrentLocation();

  // Either the LSQ is the master interface ([%memref : type]) or it talks to
  // the external memory through an MC ([MC])
  bool isMaster;
  if (parseMemRegion(parser, /*allowMC=*/true, isMaster, operands,
                     operandTypes))
    return failure();

  // Parse operand sections, in segment order
  SmallVector<int32_t> segments{isMaster ? 1 : 0};
  bool hasMC = false;
  for (StringRef keyword :
       {"start", "ctrl", "ldAddr", "stAddr", "stData", "fromMC", "end"}) {
    FailureOr<unsigned> numOprds = parseMemSection(parser, keyword, operands);
    if (failed(numOprds))
      return failure();
    if (keyword == "fromMC")
      hasMC = *numOprds != 0;
    segments.push_back(*numOprds);
  }
  if (isMaster == hasMC)
    return parser.emitError(loc)
           << "LSQ must either reference a memory region or have a load data "
              "input from an MC (fromMC), but not both";
  result.addAttribute(getOprdSegmentsAttrName(),
                      parser.getBuilder().getDenseI32ArrayAttr(segments));

  if (parseMemTypes(parser, result, loc, operands, operandTypes))
    return failure();

  // Derive the result segments from the groups
  FailureOr<SmallVector<MemGroupAttr>> groups =
      getParsedGroups(parser, result, loc);
  if (failed(groups))
    return failure();
  int32_t numLoads = 0;
  for (MemGroupAttr group : *groups)
    numLoads += group.getNumLoads();
  int32_t toMC = hasMC ? 1 : 0;
  return setParsedResultSegments(
      parser, result, loc, {numLoads, toMC, toMC, toMC, isMaster ? 1 : 0});
}

void LSQOp::print(OpAsmPrinter &p) {
  ValueRange inputs = getOperands();
  if (Value memref = getMasterMemRef()) {
    p << "[" << memref << " : " << memref.getType() << "]";
    inputs = inputs.drop_front();
  } else {
    p << "[MC]";
  }
  if (Value memStart = getMasterMemStart())
    printMemSection(p, "start", memStart);
  printMemSection(p, "ctrl", getCtrls());
  printMemSection(p, "ldAddr", getLdAddrs());
  printMemSection(p, "stAddr", getStAddrs());
  printMemSection(p, "stData", getStDatas());
  if (Value ldDataFromMC = getLdDataFromMC())
    printMemSection(p, "fromMC", ldDataFromMC);
  if (Value ctrlEnd = getMasterCtrlEnd())
    printMemSection(p, "end", ctrlEnd);
  p.printOptionalAttrDict((*this)->getAttrs(), {getOprdSegmentsAttrName(),
                                                getResSegmentsAttrName()});
  p << " : ";
  p.printFunctionalType(inputs.getTypes(), getResultTypes());
}

/// Derives the ports of an LSQ from its operand segments and groups.
static void getLSQPorts(LSQPorts &lsqPorts) {
  handshake::LSQOp lsqOp = lsqPorts.getLSQOp();
  // All LSQ groups have a control port (checked by the verifier)
  addGroupPorts(lsqPorts, lsqOp.getMemGroups(), lsqOp.getCtrlsMutable(),
                lsqOp.getLdAddrsMutable(), lsqOp.getStAddrsMutable(),
                lsqOp.getStDatasMutable(), lsqOp.getLdDatas());

  if (lsqOp.isConnectedToMC()) {
    lsqPorts.interfacePorts.push_back(
        MCLoadStorePort(lsqOp.getConnectedMC(),
                        cast<OpResult>(lsqOp.getLdAddrToMC()).getResultNumber(),
                        lsqOp.getLdDataFromMCMutable()[0].getOperandNumber()));
  }

  SmallVector<Value> ctrls(lsqOp.getCtrls());
  SmallVector<Value> addrs(lsqOp.getLdAddrs());
  llvm::append_range(addrs, lsqOp.getStAddrs());
  addrs.append({lsqOp.getLdAddrToMC(), lsqOp.getStAddrToMC()});
  SmallVector<Value> datas(lsqOp.getStDatas());
  llvm::append_range(datas, lsqOp.getLdDatas());
  datas.append({lsqOp.getStDataToMC(), lsqOp.getLdDataFromMC()});
  lsqPorts.ctrlWidth = getFirstBitwidth(ctrls);
  lsqPorts.addrWidth = getFirstBitwidth(addrs);
  lsqPorts.dataWidth = getFirstBitwidth(datas);
}

LogicalResult LSQOp::verify() {
  SmallVector<MemGroupAttr> groups = getMemGroups();
  if (failed(verifyMemGroups(*this, groups, getCtrls().size(),
                             getLdAddrs().size(), getLdDatas().size(),
                             getStAddrs().size(), getStDatas().size())))
    return failure();

  // Every group must have a control port and at least one access
  for (MemGroupAttr group : groups) {
    if (!group.getHasCtrl())
      return emitError() << "LSQ group " << group.getId()
                         << " must have a control port.";
    if (group.getAccesses().empty())
      return emitError() << "Group sizes must be stricly greater than 0.";
  }

  // The LSQ is either the master interface or connected to an MC
  bool isMaster = getMasterMemRef() != nullptr;
  if (isMaster != (getMasterMemStart() != nullptr) ||
      isMaster != (getMasterCtrlEnd() != nullptr) ||
      isMaster != (getMasterMemEnd() != nullptr))
    return emitError() << "memory region, memory start, control end, and "
                          "memory end must all be present or absent.";
  bool hasMC = isConnectedToMC();
  if (hasMC != (getLdAddrToMC() != nullptr) ||
      hasMC != (getStAddrToMC() != nullptr) ||
      hasMC != (getStDataToMC() != nullptr))
    return emitError() << "MC load data, load address, store address, and "
                          "store data ports must all be present or absent.";
  if (isMaster == hasMC)
    return emitError() << "LSQ must either be the master interface or be "
                          "connected to an MC, but not both.";

  // Signals of the same nature must have the same bitwidth
  SmallVector<Value> addrs(getLdAddrs());
  llvm::append_range(addrs, getStAddrs());
  addrs.append({getLdAddrToMC(), getStAddrToMC()});
  SmallVector<Value> datas(getStDatas());
  llvm::append_range(datas, getLdDatas());
  datas.append({getStDataToMC(), getLdDataFromMC()});
  return success(succeeded(verifySameBitwidth(
                     *this, "control ports", SmallVector<Value>(getCtrls()))) &&
                 succeeded(verifySameBitwidth(*this, "addresses", addrs)) &&
                 succeeded(verifySameBitwidth(*this, "data", datas)));
}

dynamatic::LSQPorts LSQOp::getPorts() {
  LSQPorts lsqPorts(*this);
  getLSQPorts(lsqPorts);
  return lsqPorts;
}

handshake::MemoryControllerOp LSQOp::getConnectedMC() {
  Value ldDataFromMC = getLdDataFromMC();
  if (!ldDataFromMC)
    return nullptr;
  return dyn_cast_if_present<handshake::MemoryControllerOp>(
      backtrackToMemInput(ldDataFromMC));
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
  if (auto mcOp = dyn_cast<handshake::MemoryControllerOp>((Operation *)memOp))
    return mcOp.getPorts();
  if (auto lsqOp = dyn_cast<handshake::LSQOp>((Operation *)memOp))
    return lsqOp.getPorts();
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
  return cast_if_present<handshake::LoadOp>(portOp);
}

StorePort::StorePort(handshake::StoreOp storeOp, unsigned addrInputIdx,
                     unsigned dataInputIdx)
    : MemoryPort(storeOp, {addrInputIdx, dataInputIdx}, {}, Kind::STORE) {}

handshake::StoreOp StorePort::getStoreOp() const {
  return cast_if_present<handshake::StoreOp>(portOp);
}

LSQLoadStorePort::LSQLoadStorePort(dynamatic::handshake::LSQOp lsqOp,
                                   unsigned loadAddrInputIdx,
                                   unsigned loadDataOutputIdx)
    : MemoryPort(lsqOp,
                 {loadAddrInputIdx, loadAddrInputIdx + 1, loadAddrInputIdx + 2},
                 {loadDataOutputIdx}, Kind::LSQ_LOAD_STORE) {}

handshake::LSQOp LSQLoadStorePort::getLSQOp() const {
  return cast_if_present<handshake::LSQOp>(portOp);
}

MCLoadStorePort::MCLoadStorePort(dynamatic::handshake::MemoryControllerOp mcOp,
                                 unsigned loadAddrOutputIdx,
                                 unsigned loadDataInputIdx)
    : MemoryPort(
          mcOp, {loadDataInputIdx},
          {loadAddrOutputIdx, loadAddrOutputIdx + 1, loadAddrOutputIdx + 2},
          Kind::MC_LOAD_STORE) {}

handshake::MemoryControllerOp MCLoadStorePort::getMCOp() const {
  return cast_if_present<handshake::MemoryControllerOp>(portOp);
}

//===----------------------------------------------------------------------===//
// GroupMemoryPorts
//===----------------------------------------------------------------------===//

GroupMemoryPorts::GroupMemoryPorts(ControlPort ctrlPort) : ctrlPort(ctrlPort) {}

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

//===----------------------------------------------------------------------===//
// FuncMemoryPorts (and children)
//===----------------------------------------------------------------------===//

/// Collects the memory interface's operands (resp. results) that the ports
/// refer to, in port order.
static SmallVector<Value> getPortValues(ArrayRef<const MemoryPort *> ports,
                                        Operation *memOp, bool operands) {
  SmallVector<Value> values;
  for (const MemoryPort *port : ports) {
    ArrayRef<unsigned> indices =
        operands ? port->getOprdIndices() : port->getResIndices();
    for (unsigned idx : indices)
      values.push_back(operands ? memOp->getOperand(idx)
                                : memOp->getResult(idx));
  }
  return values;
}

/// Returns pointers to all the ports (control first) of a group.
static SmallVector<const MemoryPort *>
getGroupPorts(const GroupMemoryPorts &group) {
  SmallVector<const MemoryPort *> ports;
  if (group.ctrlPort)
    ports.push_back(&*group.ctrlPort);
  for (const MemoryPort &port : group.accessPorts)
    ports.push_back(&port);
  return ports;
}

SmallVector<Value> FuncMemoryPorts::getGroupInputs(unsigned groupIdx) {
  assert(groupIdx < groups.size() && "index higher than number of groups");
  return getPortValues(getGroupPorts(groups[groupIdx]), memOp,
                       /*operands=*/true);
}

SmallVector<Value> FuncMemoryPorts::getGroupResults(unsigned groupIdx) {
  assert(groupIdx < groups.size() && "index higher than number of groups");
  return getPortValues(getGroupPorts(groups[groupIdx]), memOp,
                       /*operands=*/false);
}

SmallVector<Value> FuncMemoryPorts::getInterfacesInputs() {
  SmallVector<const MemoryPort *> ports;
  for (const MemoryPort &port : interfacePorts)
    ports.push_back(&port);
  return getPortValues(ports, memOp, /*operands=*/true);
}

SmallVector<Value> FuncMemoryPorts::getInterfacesResults() {
  SmallVector<const MemoryPort *> ports;
  for (const MemoryPort &port : interfacePorts)
    ports.push_back(&port);
  return getPortValues(ports, memOp, /*operands=*/false);
}

MCBlock::MCBlock(GroupMemoryPorts *group, unsigned blockID)
    : blockID(blockID), group(group) {}

MCPorts::MCPorts(handshake::MemoryControllerOp mcOp) : FuncMemoryPorts(mcOp) {}

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

LSQPorts::LSQPorts(handshake::LSQOp lsqOp) : FuncMemoryPorts(lsqOp) {}

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

bool CmpIOp::isSignedComparison() {
  switch (getPredicate()) {
  case CmpIPredicate::slt:
  case CmpIPredicate::sle:
  case CmpIPredicate::sgt:
  case CmpIPredicate::sge:
    return true;
  default:
    return false;
  }
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

//===----------------------------------------------------------------------===//
// Memory Controller GetOperandName and GetResultName Utilities
//===----------------------------------------------------------------------===//

/// Load/Store base signal names common to all memory interfaces
static constexpr llvm::StringLiteral MEMREF("memref"), MEM_START("memStart"),
    MEM_END("memEnd"), CTRL_END("ctrlEnd"), CTRL("ctrl"), LD_ADDR("ldAddr"),
    LD_DATA("ldData"), ST_ADDR("stAddr"), ST_DATA("stData");

static inline std::string getArrayElemName(const Twine &name, unsigned idx) {
  return name.str() + "_" + std::to_string(idx);
}

inline static StringRef getIfControlOprd(MemoryOpInterface memOp,
                                         unsigned idx) {
  if (!memOp.isMasterInterface())
    return "";
  switch (idx) {
  case 0:
    return MEMREF;
  case 1:
    return MEM_START;
  default:
    return idx == memOp->getNumOperands() - 1 ? CTRL_END : "";
  }
}

static StringRef getIfControlRes(MemoryOpInterface memOp, unsigned idx) {
  if (memOp.isMasterInterface() && idx == memOp->getNumResults() - 1)
    return MEM_END;
  return "";
}

/// Common operand naming logic for memory controllers and LSQs.
inline static std::string getMemOperandName(const FuncMemoryPorts &ports,
                                            unsigned idx) {
  // Iterate through all memory ports to find out the type of the operand
  unsigned ctrlIdx = 0, loadIdx = 0, storeIdx = 0;
  for (const GroupMemoryPorts &blockPorts : ports.groups) {
    if (blockPorts.hasControl()) {
      if (idx == blockPorts.ctrlPort->getCtrlInputIndex())
        return getArrayElemName(CTRL, ctrlIdx);
      ++ctrlIdx;
    }
    for (const MemoryPort &accessPort : blockPorts.accessPorts) {
      if (std::optional<LoadPort> loadPort = dyn_cast<LoadPort>(accessPort)) {
        if (loadPort->getAddrInputIndex() == idx)
          return getArrayElemName(LD_ADDR, loadIdx);
        ++loadIdx;
      } else {
        std::optional<StorePort> storePort = cast<StorePort>(accessPort);
        if (storePort->getAddrInputIndex() == idx)
          return getArrayElemName(ST_ADDR, storeIdx);
        if (storePort->getDataInputIndex() == idx)
          return getArrayElemName(ST_DATA, storeIdx);
        ++storeIdx;
      }
    }
  }

  return "";
}

/// Common result naming logic for memory controllers and LSQs.
static std::string getMemResultName(FuncMemoryPorts &ports, unsigned idx) {
  // Iterate through all memory ports to find out the type of the
  // operand
  unsigned loadIdx = 0;
  for (const GroupMemoryPorts &blockPorts : ports.groups) {
    for (const MemoryPort &accessPort : blockPorts.accessPorts) {
      if (std::optional<LoadPort> loadPort = dyn_cast<LoadPort>(accessPort)) {
        if (loadPort->getDataOutputIndex() == idx)
          return getArrayElemName(LD_DATA, loadIdx);
        ++loadIdx;
      }
    }
  }
  return "";
}

std::string LSQOp::getOperandName(unsigned idx) {

  assert(idx < getOperation()->getNumOperands() && "index too high");

  if (StringRef name = getIfControlOprd(*this, idx); !name.empty())
    return name.str();

  // Try to get the operand name from the regular ports
  LSQPorts lsqPorts = getPorts();
  if (std::string name = getMemOperandName(lsqPorts, idx); !name.empty())
    return name;

  // Get the operand name from a port to a memory controller
  assert(lsqPorts.connectsToMC() && "expected LSQ to connect to MC");
  assert(lsqPorts.getMCPort().getLoadDataInputIndex() == idx &&
         "unknown LSQ/MC operand");
  return "ldDataFromMC";
}

std::string LSQOp::getResultName(unsigned idx) {
  assert(idx < getOperation()->getNumResults() && "index too high");

  if (StringRef name = getIfControlRes(*this, idx); !name.empty())
    return name.str();

  // Try to get the operand name from the regular ports
  LSQPorts lsqPorts = getPorts();
  if (std::string name = getMemResultName(lsqPorts, idx); !name.empty())
    return name;

  // Get the operand name from a port to a memory controller
  assert(lsqPorts.connectsToMC() && "expected LSQ to connect to MC");
  MCLoadStorePort mcPort = lsqPorts.getMCPort();
  if (mcPort.getLoadAddrOutputIndex() == idx)
    return "ldAddrToMC";
  if (mcPort.getStoreAddrOutputIndex() == idx)
    return "stAddrToMC";
  assert(mcPort.getStoreDataOutputIndex() == idx && "unknown LSQ/MC result");
  return "stDataToMC";
}

std::string MemoryControllerOp::getOperandName(unsigned idx) {
  assert(idx < getOperation()->getNumOperands() && "index too high");

  if (StringRef name = getIfControlOprd(*this, idx); !name.empty())
    return name.str();

  // Try to get the operand name from the regular ports
  MCPorts mcPorts = getPorts();
  if (std::string name = getMemOperandName(mcPorts, idx); !name.empty())
    return name;

  // Get the operand name from a port to an LSQ
  assert(mcPorts.connectsToLSQ() && "expected MC to connect to LSQ");
  LSQLoadStorePort lsqPort = mcPorts.getLSQPort();
  if (lsqPort.getLoadAddrInputIndex() == idx)
    return getArrayElemName(LD_ADDR, mcPorts.getNumPorts<LoadPort>());
  if (lsqPort.getStoreAddrInputIndex() == idx)
    return getArrayElemName(ST_ADDR, mcPorts.getNumPorts<StorePort>());
  assert(lsqPort.getStoreDataInputIndex() == idx && "unknown MC/LSQ operand");
  return getArrayElemName(ST_DATA, mcPorts.getNumPorts<StorePort>());
}

std::string MemoryControllerOp::getResultName(unsigned idx) {
  assert(idx < getOperation()->getNumResults() && "index too high");

  if (StringRef name = getIfControlRes(*this, idx); !name.empty())
    return name.str();

  // Try to get the operand name from the regular ports
  MCPorts mcPorts = getPorts();
  if (std::string name = getMemResultName(mcPorts, idx); !name.empty())
    return name;

  // Get the operand name from a port to an LSQ
  assert(mcPorts.connectsToLSQ() && "expected MC to connect to LSQ");
  LSQLoadStorePort lsqPort = mcPorts.getLSQPort();
  assert(lsqPort.getLoadDataOutputIndex() == idx && "unknown MC/LSQ result");
  return getArrayElemName(LD_DATA, mcPorts.getNumPorts<LoadPort>());
}

#define GET_OP_CLASSES
#include "dynamatic/Dialect/Handshake/Handshake.cpp.inc"
