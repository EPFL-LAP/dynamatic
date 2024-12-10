//===- HandshakeTypes.cpp - Handshake types ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the API to express and manipulate custom Handshake types.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include <cctype>
#include <iostream>
#include <ostream>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

//===----------------------------------------------------------------------===//
// Common implementations for ControlType and ChannelType
//===----------------------------------------------------------------------===//

static constexpr llvm::StringLiteral UPSTREAM_SYMBOL("U");

/// Print the extra signals to generate an IR representation.
/// For example: [spec: i1, tag: i32]
/// This function is common to both ControlType and ChannelType.
static void printExtraSignals(AsmPrinter &odsPrinter,
                              llvm::ArrayRef<ExtraSignal> extraSignals) {
  auto printSignal = [&](const ExtraSignal &signal) {
    odsPrinter << signal.name << ": " << signal.type;
    if (!signal.downstream)
      odsPrinter << " (" << UPSTREAM_SYMBOL << ")";
  };

  // Print all signals enclosed in square brackets
  odsPrinter << "[";
  for (const ExtraSignal &signal : extraSignals.drop_back()) {
    printSignal(signal);
    odsPrinter << ", ";
  }
  printSignal(extraSignals.back());
  odsPrinter << "]";
}

/// Checks the validity of a channel's extra signals.
static LogicalResult
checkChannelExtra(function_ref<InFlightDiagnostic()> emitError,
                  const ArrayRef<ExtraSignal> &extraSignals) {
  DenseSet<StringRef> names;
  for (const ExtraSignal &extra : extraSignals) {
    if (auto [_, newName] = names.insert(extra.name); !newName) {
      return emitError() << "expected all signal names to be unique but '"
                         << extra.name << "' appears more than once";
    }
    if (!ChannelType::isSupportedSignalType(extra.type)) {
      return emitError() << "expected extra signal type to be IntegerType or "
                            "FloatType, but "
                         << extra.name << "' has type " << extra.type;
    }
    for (char c : extra.name) {
      if (!std::isalnum(c) && c != '_') {
        return emitError() << "expected only alphanumeric characters and '_' "
                              "in extra signal name, but got "
                           << extra.name;
      }
    }

    auto checkReserved = [&](StringRef reserved) -> LogicalResult {
      if (reserved == extra.name) {
        return emitError() << "'" << reserved
                           << "' is a reserved name, it cannot be used as "
                              "an extra signal name";
      }
      return success();
    };
    if (failed(checkReserved("valid")) || failed(checkReserved("ready")))
      return failure();
  }
  return success();
}

/// Parse the extra signals from the IR representation.
/// For example: [spec: i1, tag: i32]
/// This function is common to both ControlType and ChannelType.
static LogicalResult
parseExtraSignals(function_ref<InFlightDiagnostic()> emitError,
                  AsmParser &odsParser,
                  SmallVector<ExtraSignal::Storage> &extraSignalsStorage) {

  auto parseSignal = [&]() -> ParseResult {
    auto &signal = extraSignalsStorage.emplace_back();

    if (odsParser.parseKeywordOrString(&signal.name) ||
        odsParser.parseColon() || odsParser.parseType(signal.type))
      return failure();

    // Attempt to parse the optional upstream symbol
    if (!odsParser.parseOptionalLParen()) {
      std::string upstreamSymbol;
      if (odsParser.parseKeywordOrString(&upstreamSymbol) ||
          upstreamSymbol != UPSTREAM_SYMBOL || odsParser.parseRParen())
        return failure();
      signal.downstream = false;
    }
    return success();
  };

  if (odsParser.parseCommaSeparatedList(parseSignal))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// ControlType
//===----------------------------------------------------------------------===//

void ControlType::print(AsmPrinter &odsPrinter) const {
  odsPrinter << "<";
  if (!getExtraSignals().empty()) {
    printExtraSignals(odsPrinter, getExtraSignals());
  }
  odsPrinter << ">";
}

static Type parseControlAfterLSquare(AsmParser &odsParser) {
  function_ref<InFlightDiagnostic()> emitError = [&]() {
    return odsParser.emitError(odsParser.getCurrentLocation());
  };

  SmallVector<ExtraSignal::Storage> extraSignalsStorage;
  if (failed(parseExtraSignals(emitError, odsParser, extraSignalsStorage)))
    return {};

  // Parse ']'
  if (odsParser.parseRSquare())
    return {};
  // Parse literal '>'
  if (odsParser.parseGreater())
    return {};

  SmallVector<ExtraSignal> extraSignals;
  // Convert the element type of the extra signal storage list to its
  // non-storage version (these will be uniqued/allocated by ChannelType::get)
  for (const ExtraSignal::Storage &signalStorage : extraSignalsStorage)
    extraSignals.emplace_back(signalStorage);

  if (failed(checkChannelExtra(emitError, extraSignals)))
    return {};

  return ControlType::get(odsParser.getContext(), extraSignals);
}

Type ControlType::parse(AsmParser &odsParser) {
  // Parse literal '<'
  if (odsParser.parseLess())
    return {};

  if (!odsParser.parseOptionalLSquare()) {
    // Parsed '['.
    // The control has extra bits
    return parseControlAfterLSquare(odsParser);
  }

  // Parse literal '>'
  if (odsParser.parseGreater())
    return {};

  return ControlType::get(odsParser.getContext(), {});
}

Type ControlType::addExtraSignal(const ExtraSignal &signal) const {
  SmallVector<ExtraSignal> newExtraSignals(getExtraSignals());
  newExtraSignals.emplace_back(signal);
  return ControlType::get(getContext(), newExtraSignals);
}

Type ControlType::removeExtraSignal(const std::string &name) const {
  SmallVector<ExtraSignal> newExtraSignals;
  for (const ExtraSignal &signal : getExtraSignals()) {
    if (signal.name != name)
      newExtraSignals.emplace_back(signal);
  }
  return ControlType::get(getContext(), newExtraSignals);
}

//===----------------------------------------------------------------------===//
// ChannelType
//===----------------------------------------------------------------------===//

unsigned dynamatic::handshake::getHandshakeTypeBitWidth(Type type) {
  return llvm::TypeSwitch<Type, unsigned>(type)
      .Case<handshake::ControlType>([](auto) { return 0; })
      .Case<handshake::ChannelType>(
          [](ChannelType channelType) { return channelType.getDataBitWidth(); })
      .Case<IntegerType, FloatType>(
          [](Type type) { return type.getIntOrFloatBitWidth(); })
      .Default([](Type type) {
        llvm_unreachable("unsupported type");
        return 0;
      });
}

/// Checks the validity of a channel's data type.
static LogicalResult
checkChannelData(function_ref<InFlightDiagnostic()> emitError, Type dataType) {
  if (!ChannelType::isSupportedSignalType(dataType)) {
    return emitError()
           << "expected data type to be IntegerType or FloatType, but got "
           << dataType;
  }
  if (dataType.getIntOrFloatBitWidth() == 0) {
    return emitError()
           << "expected data type to have strictly positive bitwidth, but got "
           << dataType;
  }
  return success();
}

static Type parseChannelAfterLess(AsmParser &odsParser) {
  FailureOr<Type> dataType;

  function_ref<InFlightDiagnostic()> emitError = [&]() {
    return odsParser.emitError(odsParser.getCurrentLocation());
  };

  // Parse variable 'dataType'
  dataType = FieldParser<Type>::parse(odsParser);
  if (failed(dataType)) {
    odsParser.emitError(odsParser.getCurrentLocation(),
                        "failed to parse ChannelType parameter 'dataType' "
                        "which is to be a Type");
    return nullptr;
  }
  if (failed(checkChannelData(emitError, *dataType)))
    return nullptr;

  SmallVector<ExtraSignal::Storage> extraSignalsStorage;
  if (!odsParser.parseOptionalComma()) {
    // Parsed literal ','
    // The channel has extra bits

    // Parse '['
    if (odsParser.parseLSquare())
      return nullptr;

    if (failed(parseExtraSignals(emitError, odsParser, extraSignalsStorage)))
      return nullptr;

    // Parse ']'
    if (odsParser.parseRSquare())
      return nullptr;
  }

  // Parse literal '>'
  if (odsParser.parseGreater())
    return {};

  SmallVector<ExtraSignal> extraSignals;
  // Convert the element type of the extra signal storage list to its
  // non-storage version (these will be uniqued/allocated by ChannelType::get)
  for (const ExtraSignal::Storage &signalStorage : extraSignalsStorage)
    extraSignals.emplace_back(signalStorage);

  if (failed(checkChannelExtra(emitError, extraSignals)))
    return {};

  return ChannelType::get(odsParser.getContext(), *dataType, extraSignals);
}

Type ChannelType::parse(AsmParser &odsParser) {
  // Parse literal '<'
  if (odsParser.parseLess())
    return {};
  return parseChannelAfterLess(odsParser);
}

void ChannelType::print(AsmPrinter &odsPrinter) const {
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getDataType());
  if (!getExtraSignals().empty()) {
    odsPrinter << ", ";
    printExtraSignals(odsPrinter, getExtraSignals());
  }
  odsPrinter << ">";
}

ChannelType ChannelType::getAddrChannel(MLIRContext *ctx) {
  OpBuilder builder(ctx);
  return get(builder.getIntegerType(32));
}

LogicalResult ChannelType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  Type dataType,
                                  ArrayRef<ExtraSignal> extraSignals) {
  return failure(failed(checkChannelData(emitError, dataType)) ||
                 failed(checkChannelExtra(emitError, extraSignals)));
}

unsigned ChannelType::getDataBitWidth() const {
  return getDataType().getIntOrFloatBitWidth();
}

Type dynamatic::handshake::detail::jointHandshakeTypeParser(AsmParser &parser) {
  if (parser.parseOptionalLess())
    return nullptr;
  if (!parser.parseOptionalGreater()) {
    // Parsed <>: ControlType without extra bits
    return handshake::ControlType::get(parser.getContext());
  }
  if (!parser.parseOptionalLSquare()) {
    // Parsed <[: ControlType with extra bits
    return parseControlAfterLSquare(parser);
  }
  return parseChannelAfterLess(parser);
}

Type ChannelType::addExtraSignal(const ExtraSignal &signal) const {
  SmallVector<ExtraSignal> newExtraSignals(getExtraSignals());
  newExtraSignals.emplace_back(signal);
  return ChannelType::get(getDataType(), newExtraSignals);
}

Type ChannelType::removeExtraSignal(const std::string &name) const {
  SmallVector<ExtraSignal> newExtraSignals;
  for (const ExtraSignal &signal : getExtraSignals()) {
    if (signal.name != name)
      newExtraSignals.emplace_back(signal);
  }
  return ChannelType::get(getDataType(), newExtraSignals);
}

//===----------------------------------------------------------------------===//
// ExtraSignal
//===----------------------------------------------------------------------===//

ExtraSignal::Storage::Storage(StringRef name, mlir::Type type, bool downstream)
    : name(name), type(type), downstream(downstream) {}

ExtraSignal::ExtraSignal(StringRef name, mlir::Type type, bool downstream)
    : name(name), type(type), downstream(downstream) {}

ExtraSignal::ExtraSignal(const ExtraSignal::Storage &storage)
    : name(storage.name), type(storage.type), downstream(storage.downstream) {}

unsigned ExtraSignal::getBitWidth() const {
  return type.getIntOrFloatBitWidth();
}

bool dynamatic::handshake::operator==(const ExtraSignal &lhs,
                                      const ExtraSignal &rhs) {
  return lhs.name == rhs.name && lhs.type == rhs.type &&
         lhs.downstream == rhs.downstream;
}

llvm::hash_code dynamatic::handshake::hash_value(const ExtraSignal &signal) {
  return llvm::hash_combine(signal.name, signal.type, signal.downstream);
}

#include "dynamatic/Dialect/Handshake/HandshakeTypeInterfaces.cpp.inc"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.cpp.inc"
