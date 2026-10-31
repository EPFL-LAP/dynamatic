//===- FormalProperty.cpp ---------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the JSON-parsing logic for the formal properties' database.
//
//===----------------------------------------------------------------------===//

#include "experimental/Support/FormalProperty.h"
#include "dynamatic/Support/LLVM.h"
#include "experimental/Support/PropertyExpr.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <string>
#include <variant>
#include <vector>

using namespace dynamatic;

namespace {
constexpr StringLiteral ID_LIT = "id";
constexpr StringLiteral KIND_LIT = "kind";
constexpr StringLiteral TAG_LIT = "tag";
constexpr StringLiteral CHECK_LIT = "check";
constexpr StringLiteral EXPR_LIT = "expr";
} // namespace

std::optional<FormalProperty::TAG> FormalProperty::tagFromStr(StringRef s) {
  if (s == "OPT")
    return TAG::OPT;
  if (s == "INVAR")
    return TAG::INVAR;
  if (s == "ERROR")
    return TAG::ERROR;
  return std::nullopt;
}

StringRef FormalProperty::tagToStr(TAG t) {
  switch (t) {
  case TAG::OPT:
    return "OPT";
  case TAG::INVAR:
    return "INVAR";
  case TAG::ERROR:
    return "ERROR";
  }
  llvm_unreachable("unknown tag");
}

std::vector<Signal> FormalProperty::getResultValidSignals() const {
  std::vector<Signal> signals;
  expr.walk([&](const PropertyExpr &e) {
    if (const auto *signal = std::get_if<Signal>(&e.getNode()))
      if (signal->getKind() == Signal::Valid &&
          signal->getSide() == Signal::Result)
        signals.push_back(*signal);
  });
  return signals;
}

llvm::json::Value dynamatic::toJSON(const FormalProperty &property) {
  return llvm::json::Object({{ID_LIT, property.id},
                             {KIND_LIT, property.kind},
                             {TAG_LIT, FormalProperty::tagToStr(property.tag)},
                             {CHECK_LIT, property.check},
                             {EXPR_LIT, property.expr}});
}

bool dynamatic::fromJSON(const llvm::json::Value &value,
                         FormalProperty &property, llvm::json::Path path) {
  llvm::json::ObjectMapper mapper(value, path);
  std::string tagStr;
  if (!mapper || !mapper.map(ID_LIT, property.id) ||
      !mapper.map(KIND_LIT, property.kind) || !mapper.map(TAG_LIT, tagStr) ||
      !mapper.map(CHECK_LIT, property.check) ||
      !mapper.map(EXPR_LIT, property.expr))
    return false;

  std::optional<FormalProperty::TAG> tag = FormalProperty::tagFromStr(tagStr);
  if (!tag) {
    path.field(TAG_LIT).report("unknown tag");
    return false;
  }
  property.tag = *tag;
  return true;
}

LogicalResult FormalPropertyTable::addPropertiesFromJSON(StringRef filepath) {
  // Open the properties' database
  auto buffer = llvm::MemoryBuffer::getFile(filepath);
  if (!buffer) {
    llvm::errs() << "[WARNING] Failed to open property database file @ \""
                 << filepath << "\"\n";
    return failure();
  }

  // Try to parse the file as a JSON
  llvm::Expected<llvm::json::Value> value =
      llvm::json::parse((*buffer)->getBuffer());
  if (!value) {
    llvm::errs() << "Failed to parse property table @ \"" << filepath
                 << "\" as JSON.\n-> " << toString(value.takeError()) << "\n";
    return failure();
  }

  // Retrieve formal properties (see
  // docs/DeveloperGuide/DynamaticFeaturesAndOptimizations/FormalProperties.md)
  llvm::json::Path::Root jsonRoot(filepath);
  if (!fromJSON(*value, properties, llvm::json::Path(jsonRoot))) {
    jsonRoot.printErrorContext(*value, llvm::errs());
    return failure();
  }
  return success();
}
