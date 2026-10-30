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

/// Parser for the `custom<MemGroups>` Tablegen directive: the groups of a
/// memory interface, each with its accesses in program order. A group starts
/// with its basic block (if any) and its control port (if any).
/// Example: `bb0: [load(%la0)], bb1 %c0: [store(%sa0, %sd0), load(%la1)]`
static ParseResult
parseMemGroups(OpAsmParser &parser,
               SmallVectorImpl<OpAsmParser::UnresolvedOperand> &ctrl,
               SmallVectorImpl<OpAsmParser::UnresolvedOperand> &ldAddr,
               SmallVectorImpl<OpAsmParser::UnresolvedOperand> &stAddr,
               SmallVectorImpl<OpAsmParser::UnresolvedOperand> &stData,
               ArrayAttr &groups) {
  MLIRContext *ctx = parser.getContext();
  SmallVector<Attribute> groupAttrs;
  // A group, separated by commas
  // Example: `bb1 %c0: [store(%sa0, %sd0), load(%la1)]`
  do {
    // The basic block (if any)
    // Example: `bb1`
    std::optional<unsigned> block;
    StringRef blockKeyword;
    if (succeeded(parser.parseOptionalKeyword(&blockKeyword))) {
      unsigned blockID;
      if (!blockKeyword.consume_front("bb") ||
          blockKeyword.getAsInteger(10, blockID))
        return parser.emitError(parser.getCurrentLocation(),
                                "expected a basic block (e.g., bb0)");
      block = blockID;
    }

    // The control port (if any)
    // Example: `%c0`
    OpAsmParser::UnresolvedOperand ctrlOperand;
    OptionalParseResult hasCtrl = parser.parseOptionalOperand(ctrlOperand);
    if (hasCtrl.has_value()) {
      if (failed(*hasCtrl))
        return failure();
      ctrl.push_back(ctrlOperand);
    }
    if (parser.parseColon() || parser.parseLSquare())
      return failure();

    // The accesses, in program order. A group may have none, e.g., a basic
    // block whose stores all go through an LSQ
    SmallVector<MemAccessKind> accesses;
    if (succeeded(parser.parseOptionalRSquare())) {
      groupAttrs.push_back(
          MemGroupAttr::get(ctx, block, hasCtrl.has_value(), accesses));
      continue;
    }
    // An access, separated by commas. Its operands go to the segments of its
    // kind
    // Example: `store(%sa0, %sd0)`
    do {
      // The access kind
      // Example: `store`
      StringRef kind;
      if (parser.parseKeyword(&kind) || parser.parseLParen())
        return failure();
      if (kind == "load") {
        // Example: `%la1`
        if (parser.parseOperand(ldAddr.emplace_back()))
          return failure();
        accesses.push_back(MemAccessKind::Load);
      } else if (kind == "store") {
        // Example: `%sa0, %sd0`
        if (parser.parseOperand(stAddr.emplace_back()) || parser.parseComma() ||
            parser.parseOperand(stData.emplace_back()))
          return failure();
        accesses.push_back(MemAccessKind::Store);
      } else {
        return parser.emitError(parser.getCurrentLocation(),
                                "expected 'load' or 'store'");
      }
      if (parser.parseRParen())
        return failure();
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRSquare())
      return failure();

    groupAttrs.push_back(
        MemGroupAttr::get(ctx, block, hasCtrl.has_value(), accesses));
  } while (succeeded(parser.parseOptionalComma()));

  groups = ArrayAttr::get(ctx, groupAttrs);
  return success();
}

/// Printer for the `custom<MemGroups>` Tablegen directive (see
/// `parseMemGroups`).
static void printMemGroups(OpAsmPrinter &printer, Operation * /*op*/,
                           OperandRange ctrl, OperandRange ldAddr,
                           OperandRange stAddr, OperandRange stData,
                           ArrayAttr groups) {
  // The n-th load (store) over all groups is the n-th operand of the load
  // (store) segments
  unsigned ctrlIdx = 0, loadIdx = 0, storeIdx = 0;
  auto printAccess = [&](MemAccessKind kind) {
    if (kind == MemAccessKind::Load) {
      printer << "load(" << ldAddr[loadIdx++] << ")";
      return;
    }
    printer << "store(" << stAddr[storeIdx] << ", " << stData[storeIdx] << ")";
    ++storeIdx;
  };
  auto printGroup = [&](MemGroupAttr group) {
    // The basic block and the control port, separated by a space
    if (std::optional<unsigned> block = group.getBlock())
      printer << "bb" << *block << (group.getHasCtrl() ? " " : "");
    if (group.getHasCtrl())
      printer << ctrl[ctrlIdx++];
    printer << ": [";
    llvm::interleaveComma(group.getAccesses(), printer, printAccess);
    printer << "]";
  };
  llvm::interleaveComma(groups.getAsRange<MemGroupAttr>(), printer, printGroup);
}

/// Parser for the `custom<LSQMemory>` Tablegen directive: the memory of an LSQ
/// that talks to it, or `MC` for an LSQ that forwards its accesses to a memory
/// controller.
/// Example: `%mem : memref<4xi32>`
static ParseResult
parseLSQMemory(OpAsmParser &parser,
               std::optional<OpAsmParser::UnresolvedOperand> &memref,
               Type &memrefType) {
  // An LSQ connected to an MC
  // Example: `MC`
  if (succeeded(parser.parseOptionalKeyword("MC")))
    return success();
  // An LSQ connected to the memory
  // Example: `%mem : memref<4xi32>`
  memref.emplace();
  return failure(parser.parseOperand(*memref) || parser.parseColon() ||
                 parser.parseType(memrefType));
}

/// Printer for the `custom<LSQMemory>` Tablegen directive (see
/// `parseLSQMemory`).
static void printLSQMemory(OpAsmPrinter &printer, Operation * /*op*/,
                           Value memref, Type memrefType) {
  if (memref)
    printer << memref << " : " << memrefType;
  else
    printer << "MC";
}

/// Parser for the `custom<LSQTypes>` Tablegen directive: the address and data
/// types of an LSQ. All addresses (data) have the same type.
/// Example: `<i2>, <i32>`
static ParseResult parseLSQTypes(
    OpAsmParser &parser, ArrayAttr groups,
    std::optional<OpAsmParser::UnresolvedOperand> ownMemRef,
    SmallVectorImpl<Type> &ctrlTypes, SmallVectorImpl<Type> &ldAddrTypes,
    SmallVectorImpl<Type> &stAddrTypes, SmallVectorImpl<Type> &stDataTypes,
    SmallVectorImpl<Type> &ldDataTypes, Type &ownMemStartType,
    Type &ldDataFromMCType, Type &ownCtrlEndType, Type &ldAddrToMCType,
    Type &stAddrToMCType, Type &stDataToMCType, Type &ownMemEndType) {
  // The address and data types
  // Example: `<i2>, <i32>`
  Type addrType, dataType;
  if (parseHandshakeType(parser, addrType) || parser.parseComma() ||
      parseHandshakeType(parser, dataType))
    return failure();

  // The numbers of operands of each kind come from the groups. The verifier
  // checks that every group has a control port
  unsigned numCtrls = 0, numLoads = 0, numStores = 0;
  for (MemGroupAttr group : groups.getAsRange<MemGroupAttr>()) {
    numCtrls += group.getHasCtrl();
    numLoads += group.getNumAccesses(MemAccessKind::Load);
    numStores += group.getNumAccesses(MemAccessKind::Store);
  }
  Type controlType = ControlType::get(parser.getContext());
  ctrlTypes.append(numCtrls, controlType);
  ldAddrTypes.append(numLoads, addrType);
  ldDataTypes.append(numLoads, dataType);
  stAddrTypes.append(numStores, addrType);
  stDataTypes.append(numStores, dataType);
  // The signals to the memory or to an MC, depending on the LSQ's memory
  if (ownMemRef) {
    ownMemStartType = ownCtrlEndType = ownMemEndType = controlType;
  } else {
    ldDataFromMCType = stDataToMCType = dataType;
    ldAddrToMCType = stAddrToMCType = addrType;
  }
  return success();
}

/// Printer for the `custom<LSQTypes>` Tablegen directive (see
/// `parseLSQTypes`).
static void printLSQTypes(OpAsmPrinter &printer, Operation * /*op*/,
                          ArrayAttr /*groups*/, Value /*ownMemRef*/,
                          TypeRange /*ctrlTypes*/, TypeRange ldAddrTypes,
                          TypeRange stAddrTypes, TypeRange stDataTypes,
                          TypeRange ldDataTypes, Type /*ownMemStartType*/,
                          Type /*ldDataFromMCType*/, Type /*ownCtrlEndType*/,
                          Type /*ldAddrToMCType*/, Type /*stAddrToMCType*/,
                          Type /*stDataToMCType*/, Type /*ownMemEndType*/) {
  // The verifier checks that the types of each kind are uniform, and every
  // group has at least one access
  Type addrType =
      !ldAddrTypes.empty() ? ldAddrTypes.front() : stAddrTypes.front();
  Type dataType =
      !ldDataTypes.empty() ? ldDataTypes.front() : stDataTypes.front();
  printHandshakeType(printer, nullptr, addrType);
  printer << ", ";
  printHandshakeType(printer, nullptr, dataType);
}

/// Parser for the `custom<MemTypes>` Tablegen directive: the address and data
/// types of a memory interface, followed by the control type if it has control
/// ports. All addresses (data) have the same type.
/// Example: `<i2>, <i32>, <i32>`
static ParseResult parseMemTypes(
    OpAsmParser &parser, ArrayAttr groups,
    std::optional<OpAsmParser::UnresolvedOperand> lsqLdAddr,
    SmallVectorImpl<Type> &ctrlTypes, SmallVectorImpl<Type> &ldAddrTypes,
    SmallVectorImpl<Type> &stAddrTypes, SmallVectorImpl<Type> &stDataTypes,
    SmallVectorImpl<Type> &ldDataTypes, Type &lsqLdAddrType,
    Type &lsqStAddrType, Type &lsqStDataType, Type &lsqLdDataType) {
  // The address and data types
  // Example: `<i2>, <i32>`
  Type addrType, dataType;
  if (parseHandshakeType(parser, addrType) || parser.parseComma() ||
      parseHandshakeType(parser, dataType))
    return failure();

  // The numbers of operands of each kind come from the groups
  unsigned numCtrls = 0, numLoads = 0, numStores = 0;
  for (MemGroupAttr group : groups.getAsRange<MemGroupAttr>()) {
    numCtrls += group.getHasCtrl();
    numLoads += group.getNumAccesses(MemAccessKind::Load);
    numStores += group.getNumAccesses(MemAccessKind::Store);
  }
  // The control type (if there are control ports)
  // Example: `<i32>`
  if (numCtrls > 0) {
    Type ctrlType;
    if (parser.parseComma() || parseHandshakeType(parser, ctrlType))
      return failure();
    ctrlTypes.append(numCtrls, ctrlType);
  }
  ldAddrTypes.append(numLoads, addrType);
  ldDataTypes.append(numLoads, dataType);
  stAddrTypes.append(numStores, addrType);
  stDataTypes.append(numStores, dataType);
  if (lsqLdAddr) {
    lsqLdAddrType = lsqStAddrType = addrType;
    lsqStDataType = lsqLdDataType = dataType;
  }
  return success();
}

/// Printer for the `custom<MemTypes>` Tablegen directive (see
/// `parseMemTypes`).
static void printMemTypes(OpAsmPrinter &printer, Operation * /*op*/,
                          ArrayAttr /*groups*/, Value /*lsqLdAddr*/,
                          TypeRange ctrlTypes, TypeRange ldAddrTypes,
                          TypeRange stAddrTypes, TypeRange stDataTypes,
                          TypeRange ldDataTypes, Type lsqLdAddrType,
                          Type /*lsqStAddrType*/, Type lsqStDataType,
                          Type /*lsqLdDataType*/) {
  // The verifier checks that the types of each kind are uniform
  Type addrType = !ldAddrTypes.empty()   ? ldAddrTypes.front()
                  : !stAddrTypes.empty() ? stAddrTypes.front()
                                         : lsqLdAddrType;
  Type dataType = !ldDataTypes.empty()   ? ldDataTypes.front()
                  : !stDataTypes.empty() ? stDataTypes.front()
                                         : lsqStDataType;
  printHandshakeType(printer, nullptr, addrType);
  printer << ", ";
  printHandshakeType(printer, nullptr, dataType);
  if (!ctrlTypes.empty()) {
    printer << ", ";
    printHandshakeType(printer, nullptr, ctrlTypes.front());
  }
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
                     [&](Attribute attr) { return !isa<StringAttr>(attr); }))
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
  if (mlir::function_interface_impl::parseFunctionSignatureWithArguments(
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
  mlir::call_interface_impl::addArgAndResultAttrs(
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
    DictionaryAttr attributes, ::mlir::PropertyRef properties,
    mlir::RegionRange regions,
    SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  ConstantOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto attr = cast<TypedAttr>(adaptor.getValue());

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
}

//===----------------------------------------------------------------------===//
// MemoryControllerOp
//===----------------------------------------------------------------------===//

static handshake::ChannelType wrapChannel(Type type) {
  assert(ChannelType::isSupportedSignalType(type) && "unsupported data type");
  return handshake::ChannelType::get(type);
}

void MemoryControllerOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                               Value memRef, Value memStart, ValueRange ctrl,
                               ValueRange ldAddr, ValueRange stAddr,
                               ValueRange stData, Value lsqLdAddr,
                               Value lsqStAddr, Value lsqStData, Value ctrlEnd,
                               ArrayAttr groups) {
  // The data results have the type of the memory's elements
  Type dataType =
      wrapChannel(cast<MemRefType>(memRef.getType()).getElementType());
  SmallVector<Type> ldDataTypes(ldAddr.size(), dataType);
  Type lsqLdDataType = lsqLdAddr ? dataType : Type();
  build(odsBuilder, odsState, ldDataTypes, lsqLdDataType,
        handshake::ControlType::get(odsBuilder.getContext()), memRef, memStart,
        ctrl, ldAddr, stAddr, stData, lsqLdAddr, lsqStAddr, lsqStData, ctrlEnd,
        groups);
}

/// Returns the memory access operation that produces \p input, looking through
/// units with a single operand (e.g., buffers, forks, and extensions), or
/// nullptr if there is none. The port of a memory interface is identified by
/// its operand segment, so this only serves the users of the port's operation.
template <typename AccessOp>
static AccessOp findAccessOp(Value input) {
  Operation *op = input.getDefiningOp();
  while (op && !isa<AccessOp>(op) && op->getNumOperands() == 1)
    op = op->getOperand(0).getDefiningOp();
  return dyn_cast_if_present<AccessOp>(op);
}

/// Derives the ports of a memory controller from its operand segments and
/// groups. Fails if the bitwidths of the ports of the same kind differ.
static LogicalResult getMCPorts(MCPorts &mcPorts) {
  handshake::MemoryControllerOp mcOp = mcPorts.getMCOp();
  // Index of the first operand (result) of each segment
  auto operandStart = [&](unsigned segment) {
    return mcOp.getODSOperandIndexAndLength(segment).first;
  };
  unsigned ctrlStart = operandStart(2), ldAddrStart = operandStart(3),
           stAddrStart = operandStart(4), stDataStart = operandStart(5);
  unsigned ldDataStart = mcOp.getODSResultIndexAndLength(0).first;

  unsigned ctrlIdx = 0, loadIdx = 0, storeIdx = 0;
  for (MemGroupAttr group : mcOp.getGroups().getAsRange<MemGroupAttr>()) {
    if (group.getHasCtrl()) {
      Value ctrl = mcOp.getCtrl()[ctrlIdx];
      if (failed(checkAndSetBitwidth(ctrl, mcPorts.ctrlWidth)))
        return failure();
      mcPorts.groups.emplace_back(
          ControlPort(ctrl.getDefiningOp(), ctrlStart + ctrlIdx++));
    } else {
      mcPorts.groups.emplace_back();
    }
    GroupMemoryPorts &groupPorts = mcPorts.groups.back();

    for (MemAccessKind kind : group.getAccesses()) {
      if (kind == MemAccessKind::Load) {
        Value addr = mcOp.getLdAddr()[loadIdx];
        if (failed(checkAndSetBitwidth(addr, mcPorts.addrWidth)) ||
            failed(checkAndSetBitwidth(mcOp.getLdData()[loadIdx],
                                       mcPorts.dataWidth)))
          return failure();
        groupPorts.accessPorts.push_back(
            LoadPort(findAccessOp<handshake::LoadOp>(addr),
                     ldAddrStart + loadIdx, ldDataStart + loadIdx));
        ++loadIdx;
      } else {
        Value addr = mcOp.getStAddr()[storeIdx];
        if (failed(checkAndSetBitwidth(addr, mcPorts.addrWidth)) ||
            failed(checkAndSetBitwidth(mcOp.getStData()[storeIdx],
                                       mcPorts.dataWidth)))
          return failure();
        groupPorts.accessPorts.push_back(
            StorePort(findAccessOp<handshake::StoreOp>(addr),
                      stAddrStart + storeIdx, stDataStart + storeIdx));
        ++storeIdx;
      }
    }
  }

  // The accesses of an LSQ forwarded to the memory
  if (Value lsqLdAddr = mcOp.getLsqLdAddr()) {
    if (failed(checkAndSetBitwidth(lsqLdAddr, mcPorts.addrWidth)) ||
        failed(checkAndSetBitwidth(mcOp.getLsqStAddr(), mcPorts.addrWidth)) ||
        failed(checkAndSetBitwidth(mcOp.getLsqStData(), mcPorts.dataWidth)) ||
        failed(checkAndSetBitwidth(mcOp.getLsqLdData(), mcPorts.dataWidth)))
      return failure();
    mcPorts.interfacePorts.push_back(
        LSQLoadStorePort(findAccessOp<handshake::LSQOp>(lsqLdAddr),
                         operandStart(6), operandStart(7), operandStart(8),
                         mcOp.getODSResultIndexAndLength(1).first));
  }
  return success();
}

LogicalResult MemoryControllerOp::verify() {
  ArrayAttr groups = getGroups();
  if (groups.empty())
    return emitError() << "Memory controller must have at least one group.";

  // Each group is a distinct basic block, with a control port if it stores
  DenseSet<unsigned> uniqueBlocks;
  unsigned numCtrls = 0, numLoads = 0, numStores = 0;
  for (MemGroupAttr group : groups.getAsRange<MemGroupAttr>()) {
    std::optional<unsigned> block = group.getBlock();
    if (!block)
      return emitError() << "Every group of a memory controller must be a "
                            "basic block.";
    if (!uniqueBlocks.insert(*block).second)
      return emitError() << "Block IDs must be unique but ID " << *block
                         << " was found at least twice in the list.";
    // A block whose stores go through an LSQ has a control port without
    // stores of its own
    unsigned groupStores = group.getNumAccesses(MemAccessKind::Store);
    if (groupStores > 0 && !group.getHasCtrl())
      return emitError() << "Block " << *block
                         << " stores, so it must have a control port.";
    numCtrls += group.getHasCtrl();
    numLoads += group.getNumAccesses(MemAccessKind::Load);
    numStores += groupStores;
  }

  // The segments match the groups
  if (getCtrl().size() != numCtrls)
    return emitError() << "Expected " << numCtrls << " control ports, but got "
                       << getCtrl().size() << ".";
  if (getLdAddr().size() != numLoads || getLdData().size() != numLoads)
    return emitError() << "Expected " << numLoads
                       << " load address operands and data results.";
  if (getStAddr().size() != numStores || getStData().size() != numStores)
    return emitError() << "Expected " << numStores
                       << " store address and data operands.";

  // The ports to an LSQ are all present or all absent
  unsigned numLSQPorts = (bool)getLsqLdAddr() + (bool)getLsqStAddr() +
                         (bool)getLsqStData() + (bool)getLsqLdData();
  if (numLSQPorts != 0 && numLSQPorts != 4)
    return emitError() << "The ports to an LSQ must all be present or all be "
                          "absent.";

  // Checks the bitwidths of the ports
  MCPorts mcPorts(*this);
  return getMCPorts(mcPorts);
}

dynamatic::MCPorts MemoryControllerOp::getPorts() {
  MCPorts mcPorts(*this);
  if (failed(getMCPorts(mcPorts)))
    assert(false && "failed to identify memory ports");
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

void LSQOp::build(OpBuilder &odsBuilder, OperationState &odsState, Value memref,
                  Value memStart, ValueRange ctrl, ValueRange ldAddr,
                  ValueRange stAddr, ValueRange stData, Value ctrlEnd,
                  ArrayAttr groups) {
  // The data results have the type of the memory's elements
  Type dataType =
      wrapChannel(cast<MemRefType>(memref.getType()).getElementType());
  SmallVector<Type> ldDataTypes(ldAddr.size(), dataType);
  build(odsBuilder, odsState, ldDataTypes, /*ldAddrToMC=*/Type(),
        /*stAddrToMC=*/Type(), /*stDataToMC=*/Type(),
        handshake::ControlType::get(odsBuilder.getContext()), memref, memStart,
        ctrl, ldAddr, stAddr, stData, /*ldDataFromMC=*/Value(), ctrlEnd,
        groups);
}

void LSQOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                  handshake::MemoryControllerOp mcOp, ValueRange ctrl,
                  ValueRange ldAddr, ValueRange stAddr, ValueRange stData,
                  ArrayAttr groups) {
  // The data results have the type of the memory's elements, and the accesses
  // forwarded to the MC have the address and data types. The LSQ is a slave
  // interface in this case (the MC is the master), so it has no memory, start,
  // or end signals. The MC's load data result is connected later on.
  MLIRContext *ctx = odsBuilder.getContext();
  Type dataType = wrapChannel(mcOp.getMemRefType().getElementType());
  Type addrType = handshake::ChannelType::getAddrChannel(ctx);
  SmallVector<Type> ldDataTypes(ldAddr.size(), dataType);
  build(odsBuilder, odsState, ldDataTypes, addrType, addrType, dataType,
        /*ownMemEnd=*/Type(), /*ownMemRef=*/Value(), /*ownMemStart=*/Value(),
        ctrl, ldAddr, stAddr, stData, mcOp.getLsqLdData(),
        /*ownCtrlEnd=*/Value(), groups);
}

/// Derives the ports of an LSQ from its operand segments and groups. Fails if
/// the bitwidths of the ports of the same kind differ.
static LogicalResult getLSQPorts(LSQPorts &lsqPorts) {
  handshake::LSQOp lsqOp = lsqPorts.getLSQOp();
  // Index of the first operand (result) of each segment
  auto operandStart = [&](unsigned segment) {
    return lsqOp.getODSOperandIndexAndLength(segment).first;
  };
  auto resultStart = [&](unsigned segment) {
    return lsqOp.getODSResultIndexAndLength(segment).first;
  };
  unsigned ctrlStart = operandStart(2), ldAddrStart = operandStart(3),
           stAddrStart = operandStart(4), stDataStart = operandStart(5);
  unsigned ldDataStart = resultStart(0);

  unsigned ctrlIdx = 0, loadIdx = 0, storeIdx = 0;
  for (MemGroupAttr group : lsqOp.getGroups().getAsRange<MemGroupAttr>()) {
    Value ctrl = lsqOp.getCtrl()[ctrlIdx];
    if (failed(checkAndSetBitwidth(ctrl, lsqPorts.ctrlWidth)))
      return failure();
    lsqPorts.groups.emplace_back(
        ControlPort(ctrl.getDefiningOp(), ctrlStart + ctrlIdx++));
    GroupMemoryPorts &groupPorts = lsqPorts.groups.back();

    for (MemAccessKind kind : group.getAccesses()) {
      if (kind == MemAccessKind::Load) {
        Value addr = lsqOp.getLdAddr()[loadIdx];
        if (failed(checkAndSetBitwidth(addr, lsqPorts.addrWidth)) ||
            failed(checkAndSetBitwidth(lsqOp.getLdData()[loadIdx],
                                       lsqPorts.dataWidth)))
          return failure();
        groupPorts.accessPorts.push_back(
            LoadPort(findAccessOp<handshake::LoadOp>(addr),
                     ldAddrStart + loadIdx, ldDataStart + loadIdx));
        ++loadIdx;
      } else {
        Value addr = lsqOp.getStAddr()[storeIdx];
        if (failed(checkAndSetBitwidth(addr, lsqPorts.addrWidth)) ||
            failed(checkAndSetBitwidth(lsqOp.getStData()[storeIdx],
                                       lsqPorts.dataWidth)))
          return failure();
        groupPorts.accessPorts.push_back(
            StorePort(findAccessOp<handshake::StoreOp>(addr),
                      stAddrStart + storeIdx, stDataStart + storeIdx));
        ++storeIdx;
      }
    }
  }

  // The accesses forwarded to an MC, whose load data comes back to the LSQ
  if (Value ldDataFromMC = lsqOp.getLdDataFromMC()) {
    if (failed(checkAndSetBitwidth(ldDataFromMC, lsqPorts.dataWidth)) ||
        failed(
            checkAndSetBitwidth(lsqOp.getLdAddrToMC(), lsqPorts.addrWidth)) ||
        failed(
            checkAndSetBitwidth(lsqOp.getStAddrToMC(), lsqPorts.addrWidth)) ||
        failed(checkAndSetBitwidth(lsqOp.getStDataToMC(), lsqPorts.dataWidth)))
      return failure();
    lsqPorts.interfacePorts.push_back(MCLoadStorePort(
        findAccessOp<handshake::MemoryControllerOp>(ldDataFromMC),
        resultStart(1), operandStart(6)));
  }
  return success();
}

LogicalResult LSQOp::verify() {
  ArrayAttr groups = getGroups();
  if (groups.empty())
    return emitError() << "LSQ needs to have at least one group.";

  // Every group has a control port and at least one access
  unsigned numLoads = 0, numStores = 0;
  for (MemGroupAttr group : groups.getAsRange<MemGroupAttr>()) {
    if (!group.getHasCtrl())
      return emitError() << "Every LSQ group must have a control port.";
    if (group.getAccesses().empty())
      return emitError() << "Every LSQ group must have at least one access.";
    numLoads += group.getNumAccesses(MemAccessKind::Load);
    numStores += group.getNumAccesses(MemAccessKind::Store);
  }

  // The segments match the groups
  if (getCtrl().size() != groups.size())
    return emitError() << "Expected " << groups.size()
                       << " control ports, but got " << getCtrl().size() << ".";
  if (getLdAddr().size() != numLoads || getLdData().size() != numLoads)
    return emitError() << "Expected " << numLoads
                       << " load address operands and data results.";
  if (getStAddr().size() != numStores || getStData().size() != numStores)
    return emitError() << "Expected " << numStores
                       << " store address and data operands.";

  // An LSQ either talks to the memory (with its own memory, start, and end
  // signals) or forwards its accesses to an MC
  bool ownsMemory =
      getOwnMemRef() && getOwnMemStart() && getOwnCtrlEnd() && getOwnMemEnd();
  bool anyOwn =
      getOwnMemRef() || getOwnMemStart() || getOwnCtrlEnd() || getOwnMemEnd();
  bool forwardsToMC = getLdDataFromMC() && getLdAddrToMC() && getStAddrToMC() &&
                      getStDataToMC();
  bool anyToMC = getLdDataFromMC() || getLdAddrToMC() || getStAddrToMC() ||
                 getStDataToMC();
  if (!((ownsMemory && !anyToMC) || (forwardsToMC && !anyOwn)))
    return emitError() << "An LSQ must either have its own memory, start, and "
                          "end signals, or all ports to an MC.";

  // Checks the bitwidths of the ports
  LSQPorts lsqPorts(*this);
  return getLSQPorts(lsqPorts);
}

dynamatic::LSQPorts LSQOp::getPorts() {
  LSQPorts lsqPorts(*this);
  if (failed(getLSQPorts(lsqPorts)))
    assert(false && "failed to identify memory ports");
  return lsqPorts;
}

handshake::MemoryControllerOp LSQOp::getConnectedMC() {
  Value stDataToMC = getStDataToMC();
  if (!stDataToMC || stDataToMC.use_empty())
    return nullptr;
  return dyn_cast<handshake::MemoryControllerOp>(
      *stDataToMC.getUsers().begin());
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
    : MemoryPort(storeOp, {addrInputIdx, addrInputIdx + 1}, {}, Kind::STORE) {}

StorePort::StorePort(handshake::StoreOp storeOp, unsigned addrInputIdx,
                     unsigned dataInputIdx)
    : MemoryPort(storeOp, {addrInputIdx, dataInputIdx}, {}, Kind::STORE) {}

handshake::StoreOp StorePort::getStoreOp() const {
  return cast<handshake::StoreOp>(portOp);
}

LSQLoadStorePort::LSQLoadStorePort(dynamatic::handshake::LSQOp lsqOp,
                                   unsigned loadAddrInputIdx,
                                   unsigned loadDataOutputIdx)
    : MemoryPort(lsqOp,
                 {loadAddrInputIdx, loadAddrInputIdx + 1, loadAddrInputIdx + 2},
                 {loadDataOutputIdx}, Kind::LSQ_LOAD_STORE) {}

LSQLoadStorePort::LSQLoadStorePort(dynamatic::handshake::LSQOp lsqOp,
                                   unsigned loadAddrInputIdx,
                                   unsigned storeAddrInputIdx,
                                   unsigned storeDataInputIdx,
                                   unsigned loadDataOutputIdx)
    : MemoryPort(lsqOp,
                 {loadAddrInputIdx, storeAddrInputIdx, storeDataInputIdx},
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
    DictionaryAttr attributes, ::mlir::PropertyRef properties,
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
                         ::mlir::PropertyRef properties,
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
      cast<ChannelType>(operands[0].getType()).getExtraSignals()));
  return success();
}

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

LogicalResult
CmpIOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                         ValueRange operands, DictionaryAttr attributes,
                         ::mlir::PropertyRef properties,
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
      cast<ChannelType>(operands[0].getType()).getExtraSignals()));
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
  // trunci(trunci(x)) -> trunci(x)
  if (auto defTruncOp = getIn().getDefiningOp<TruncIOp>()) {
    // Bypass the preceeding truncation operation
    getInMutable().assign(defTruncOp.getIn());
    return getResult();
  }

  // trunci(ext(x)) -> trunci(x) when x is wider than the result, or x when it
  // has the result's type
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
    // Bypass the preceeding extension operation and the truncation, if the
    // extension's operand has the type of the truncation's result. A narrower
    // operand would need a narrower extension, which a fold cannot create
    if (src.getType() == getOut().getType())
      return src;
  }

  // trunci(x) -> x when x has the result's width: the trunc is a no-op
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
