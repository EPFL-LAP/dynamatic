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
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include <set>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

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

static constexpr llvm::StringLiteral UPSTREAM_SYMBOL("U");

static Type parseChannelAfterLess(AsmParser &odsParser) {
  FailureOr<Type> dataType;
  FailureOr<SmallVector<ExtraSignal::Storage>> extraSignalsStorage;

  // Parse variable 'dataType'
  dataType = FieldParser<Type>::parse(odsParser);
  if (failed(dataType)) {
    odsParser.emitError(odsParser.getCurrentLocation(),
                        "failed to parse ChannelType parameter 'dataType' "
                        "which is to be a Type");
    return nullptr;
  }
  if (!ChannelType::isSupportedSignalType(*dataType)) {
    odsParser.emitError(odsParser.getCurrentLocation(),
                        "failed to parse ChannelType parameter 'dataType' "
                        "which must be IntegerType or FloatType");
    return nullptr;
  }

  // Parse variable 'extraSignals'
  std::set<std::string> extraNames;
  extraSignalsStorage = [&]() -> FailureOr<SmallVector<ExtraSignal::Storage>> {
    SmallVector<ExtraSignal::Storage> storage;

    if (!odsParser.parseOptionalComma()) {
      auto parseSignal = [&]() -> ParseResult {
        auto &signal = storage.emplace_back();

        // Parse name and check for uniqueness
        if (odsParser.parseKeywordOrString(&signal.name))
          return failure();
        if (auto [_, newName] = extraNames.insert(signal.name); !newName) {
          odsParser.emitError(
              odsParser.getCurrentLocation(),
              "duplicated extra signal name, signal names must be unique");
          return failure();
        }

        // Parse colon and type and check type legality
        if (odsParser.parseColon() || odsParser.parseType(signal.type))
          return failure();
        if (!ChannelType::isSupportedSignalType(signal.type)) {
          odsParser.emitError(odsParser.getCurrentLocation(),
                              "failed to parse extra signal type which must be "
                              "IntegerType or FloatType");
          return failure();
        }

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

      if (odsParser.parseLSquare() ||
          odsParser.parseCommaSeparatedList(parseSignal) ||
          odsParser.parseRSquare())
        return failure();
    }

    return storage;
  }();
  if (failed(extraSignalsStorage)) {
    odsParser.emitError(odsParser.getCurrentLocation(),
                        "failed to parse ChannelType parameter 'extraSignals' "
                        "which is to be a ArrayRef<ExtraSignal>");
    return nullptr;
  }

  // Parse literal '>'
  if (odsParser.parseGreater())
    return {};

  // Convert the element type of the extra signal storage list to its
  // non-storage version (these will be uniqued/allocated by ChannelType::get)
  SmallVector<ExtraSignal> extraSignals;
  for (const ExtraSignal::Storage &signalStorage : *extraSignalsStorage)
    extraSignals.emplace_back(signalStorage);

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
    auto printSignal = [&](const ExtraSignal &signal) {
      odsPrinter << signal.name << ": " << signal.type;
      if (!signal.downstream)
        odsPrinter << " (" << UPSTREAM_SYMBOL << ")";
    };

    // Print all signals enclosed in square brackets
    odsPrinter << ", [";
    for (const ExtraSignal &signal : getExtraSignals().drop_back()) {
      printSignal(signal);
      odsPrinter << ", ";
    }
    printSignal(getExtraSignals().back());
    odsPrinter << "]";
    ;
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
  if (!isSupportedSignalType(dataType)) {
    return emitError()
           << "expected data type to be IntegerType or FloatType, but got "
           << dataType;
  }

  DenseSet<StringRef> names;
  for (const ExtraSignal &signal : extraSignals) {
    if (auto [_, newName] = names.insert(signal.name); !newName) {
      return emitError() << "expected all signal names to be unique but '"
                         << signal.name << "' appears more than once";
    }
    if (!isSupportedSignalType(signal.type)) {
      return emitError() << "expected extra signal type to be IntegerType or "
                            "FloatType, but "
                         << signal.name << "' has type " << signal.type;
    }
  }
  return success();
}

unsigned ChannelType::getNumDownstreamExtraSignals() const {
  return llvm::count_if(getExtraSignals(), [](const ExtraSignal &extra) {
    return extra.downstream;
  });
}

unsigned ChannelType::getDataBitWidth() const {
  return getDataType().getIntOrFloatBitWidth();
}

Type dynamatic::handshake::detail::jointHandshakeTypeParser(AsmParser &parser) {
  if (parser.parseOptionalLess())
    return nullptr;
  if (!parser.parseOptionalGreater())
    return handshake::ControlType::get(parser.getContext());
  return parseChannelAfterLess(parser);
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

#include "dynamatic/Dialect/Handshake/HandshakeTypes.cpp.inc"
