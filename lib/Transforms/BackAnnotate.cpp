//===- BackAnnotate.cpp - Back-annotate IR from JSON input ------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the back-annotation pass. To add support for a new attribute type:
//
// 1. Add a `llvm::StringLiteral` like `ATTR_TYPE_BUFFERING` with the name of
//    the new attribute type.
// 2. Create a `fromJSON` function similar to
//    ```cpp
//    static bool fromJSON(const json::Value &value,
//                         handshake::ChannelBufPropsAttr &attr,
//                         json::Path path,
//                         MLIRContext *ctx);
//    ```
//    which creates an instance of the new attribute from the data contained
//    under the "attribute-data" key of every annotation refering to the new
//    attribute type (replace the `handshake::ChannelBufPropsAttr` argument type
//    with the associated MLIR attribute type).
// 3. At the end of the `parseOpAnnotations` and/or `parseOprdAnnotations`
//    methods, add a check for the new attribute type that calls, respectively,
//    `setOpAttribute` or `setOprdAttribute` with the associated MLIR attribute
//    type as template parameter.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BackAnnotate.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/TimingModels.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include <fstream>
#include <functional>
#include <memory>
#include <unordered_map>

using namespace llvm;
using namespace mlir;
using namespace circt;
using namespace dynamatic;

/// Recognized keys in back-annotation files (excluded attribute-specific data).
static constexpr StringLiteral KEY_OPERATIONS("operations"),
    KEY_OPERANDS("operands"), KEY_OPNAME("operation-name"),
    KEY_OPRD_IDX("operand-idx"), KEY_ATTR_NAME("attribute-name"),
    KEY_ATTR_TYPE("attribute-type"), KEY_DATA("attribute-data");

/// Recognized attribute types (add new ones here, and a "fromJSON" parsing
/// function to go along).
static constexpr StringLiteral ATTR_TYPE_BUFFERING("buffering-properties");

/// JSON path errors.
static constexpr StringLiteral
    ERR_EXPECTED_OPERATIONS("expected \"operations\" key"),
    ERR_EXPECTED_OPERANDS("expected \"operands\" key"),
    ERR_EXPECTED_OPNAME("expected \"operation-name\" key"),
    ERR_EXPECTED_OPRD_IDX("expected \"operand-idx\" key"),
    ERR_EXPECTED_ATTR_NAME("expected \"attribute-name\" key"),
    ERR_EXPECTED_ATTR_TYPE("expected \"attribute-type\" key"),
    ERR_EXPECTED_DATA("expected \"attribute-data\" key"),
    ERR_EXPECTED_OBJECT("expected value to be JSON object"),
    ERR_EXPECTED_ARRAY("expected value to be JSON array"),
    ERR_EXPECTED_OP("expected operation to exist in the IR"),
    ERR_EXPECTED_OPRD(
        "expected operand index to be strictly less than number of "
        "operation operands"),
    ERR_UNKNOWN_KEY("unknown key");

/// Deserializes a JSON value under a provided JSON key into a value of the
/// template type. Returns false if the key does not exist of if the value
/// cannot be deserialized to the template type; otherwise, returns true.
template <typename T>
static bool fromJSONUnderKey(const json::Object &object, StringRef key,
                             T &value, json::Path path, StringLiteral errKey) {
  const json::Value *jsonValue = object.get(key);
  if (!jsonValue) {
    path.report(errKey);
    return false;
  }
  return json::fromJSON(*jsonValue, value, path);
}

/// Deserializes a JSON value under a provided JSON key into a value of the
/// template type if the key exists. Returns false if the key exists but the
/// value cannot be deserialized to the template type; otherwise, returns true.
template <typename T>
static bool fromJSONIfPresent(const json::Object &object, StringRef key,
                              T &value, json::Path path) {
  if (const json::Value *jsonValue = object.get(key))
    return json::fromJSON(*jsonValue, value, path.field(key));
  return true;
}

/// Returns the array of annotations (for operations or operands) located
/// directly under the top-level JSON value at the provided JSON key. Returns
/// nullptr if the array does not exist.
static const json::Array *getAnnotationArray(const json::Value &topValue,
                                             StringRef key, json::Path path,
                                             StringLiteral errKey) {
  // Top-level value must be an object
  const json::Object *topObject = topValue.getAsObject();
  if (!topObject) {
    path.report(ERR_EXPECTED_OBJECT);
    return nullptr;
  }

  // Look for the key at the top-level
  const json::Value *opsValue = topObject->get(key);
  json::Path opsPath = path.field(key);
  if (!opsValue) {
    opsPath.report(errKey);
    return nullptr;
  }

  // The value at the key must be an array
  const json::Array *opsArray = opsValue->getAsArray();
  if (!opsArray) {
    opsPath.field(key).report(ERR_EXPECTED_ARRAY);
    return nullptr;
  }
  return opsArray;
}

/// Allowed keys for "buffering-properties" attribute type
static const StringLiteral MINIMUM_TRANS("minimum-trans"),
    MAXIMUM_TRANS("maximum-trans"), MINIMUM_OPAQUE("minimum-opaque"),
    MAXIMUM_OPAQUE("maximum-opaque"), INPUT_DELAY("input-delay"),
    OUTPUT_DELAY("output-delay"), UNBUF_DELAY("unbuf-delay");

/// Deserializes a JSON value into a handshake::ChannelBufPropsAttr. See
/// ::llvm::json::Value's documentation for a longer description of this
/// function's behavior.
static bool fromJSON(const json::Value &value,
                     handshake::ChannelBufPropsAttr &attr, json::Path path,
                     MLIRContext *ctx) {
  // Under the "attribute-data" key must be a JSON object
  const json::Object *data = value.getAsObject();
  if (!data) {
    path.report(ERR_EXPECTED_OBJECT);
    return false;
  }

  ChannelBufProps props;
  // Map all supported keys to a callback to set the corresponding field in the
  // channel buffering properties
  std::map<StringRef, std::function<bool()>> keys;
  keys[MINIMUM_TRANS] = [&]() {
    return fromJSONIfPresent(*data, MINIMUM_TRANS, props.minTrans, path);
  };
  keys[MAXIMUM_TRANS] = [&]() {
    return fromJSONIfPresent(*data, MAXIMUM_TRANS, props.maxTrans, path);
  };
  keys[MINIMUM_OPAQUE] = [&]() {
    return fromJSONIfPresent(*data, MINIMUM_OPAQUE, props.minOpaque, path);
  };
  keys[MAXIMUM_OPAQUE] = [&]() {
    return fromJSONIfPresent(*data, MAXIMUM_OPAQUE, props.maxOpaque, path);
  };
  keys[INPUT_DELAY] = [&]() {
    return fromJSONIfPresent(*data, INPUT_DELAY, props.inDelay, path);
  };
  keys[OUTPUT_DELAY] = [&]() {
    return fromJSONIfPresent(*data, OUTPUT_DELAY, props.outDelay, path);
  };
  keys[UNBUF_DELAY] = [&]() {
    return fromJSONIfPresent(*data, UNBUF_DELAY, props.delay, path);
  };

  // Iterate over all keys in the data
  for (auto &[key, _] : *data) {
    if (auto callback = keys.find(key.str()); callback != keys.end()) {
      // Supported key
      if (!callback->second())
        return false;
    } else {
      // Unsupported key
      path.field(key.str()).report(ERR_UNKNOWN_KEY);
      return false;
    }
  }

  attr = handshake::ChannelBufPropsAttr::get(ctx, props);
  return true;
}

namespace {

/// Driver for the back-annotation pass. Opens and parses the JSON file
/// containing back-annotations at the provided filepath, then attempts to set
/// attributes on operations and operands referenced by the file.
struct BackAnnotatePass
    : public dynamatic::impl::BackAnnotateBase<BackAnnotatePass> {

  BackAnnotatePass(const std::string &filepath) { this->filepath = filepath; }

  void runDynamaticPass() override {
    // Open the back-annotation file
    std::ifstream inputFile(filepath);
    if (!inputFile.is_open()) {
      llvm::errs() << "Failed to open timing database\n";
      return signalPassFailure();
    }

    // Read the JSON content from the file and into a string
    std::string jsonString;
    std::string line;
    while (std::getline(inputFile, line))
      jsonString += line;

    // Try to parse the string as a JSON
    llvm::Expected<json::Value> value = json::parse(jsonString);
    if (!value) {
      llvm::errs() << "Failed to parse back-annotation file at \"" << filepath
                   << "\"\n";
      return signalPassFailure();
    }

    // Parse operation annotations and operand annotations
    json::Path::Root jsonRoot(filepath);
    if (failed(parseOpAnnotations(*value, json::Path(jsonRoot))) ||
        failed(parseOprdAnnotations(*value, json::Path(jsonRoot)))) {
      jsonRoot.printErrorContext(*value, llvm::errs());
      return signalPassFailure();
    }
  };

private:
  /// Parses operation annotations from the JSON and sets attributes on
  /// operations as requested. Fails if the JSON is badly formatted or if a
  /// referenced operation does not exist in the IR, succeeds otherwise.
  LogicalResult parseOpAnnotations(const json::Value &topValue,
                                   json::Path path);

  /// Parses operand annotations from the JSON and sets attributes on
  /// operands as requested. Fails if the JSON is badly formatted or if a
  /// referenced operand does not exist in the IR, succeeds otherwise.
  LogicalResult parseOprdAnnotations(const json::Value &topValue,
                                     json::Path path);

  /// Looks for the operation referenced by the JSON object in the IR. `object`
  /// must contain a key "operation-name" whose value is the unique operation
  /// name to look for. Reports an error and returns nullptr on failure.
  Operation *findOperation(const json::Object &object, json::Path path);

  /// Sets an attribute of the template type and of the provided name on the
  /// operation. The `value` should be the "attribute-data" key's value in the
  /// JSON file.
  template <typename Attr>
  LogicalResult setOpAttribute(Operation *op, StringRef attrName,
                               const json::Value &value, json::Path path);

  /// Sets an attribute of the template type on the operand (internally stored
  /// on the owning operation). The attribute name is determined automatically
  /// based on the attribute type's container attribute type. The `value` should
  /// be the "attribute-data" key's value in the JSON file.
  template <typename Attr>
  LogicalResult setOprdAttribute(OpOperand &oprd, const json::Value &value,
                                 json::Path path);
};

} // namespace

LogicalResult BackAnnotatePass::parseOpAnnotations(const json::Value &topValue,
                                                   json::Path path) {
  // Try to get the "operations" array
  const json::Array *opsArray = getAnnotationArray(
      topValue, KEY_OPERATIONS, path, ERR_EXPECTED_OPERATIONS);
  if (!opsArray)
    return failure();

  // Initialize the path
  json::Path arrayPath = path.field(KEY_OPERATIONS);

  // Loop over the operation annotations
  for (auto [idx, opAnnotation] : llvm::enumerate(*opsArray)) {
    json::Path opPath = arrayPath.index(idx);

    // Every operation annotation must be an object
    const json::Object *opObject = opAnnotation.getAsObject();
    if (!opObject) {
      opPath.report(ERR_EXPECTED_OBJECT);
      return failure();
    }

    // Try to find the operation with the referenced name in the IR
    Operation *op = findOperation(*opObject, opPath);
    if (!op)
      return failure();

    // Operation annotation must reference an attribute type and name
    std::string attrType, attrName;
    if (!fromJSONUnderKey(*opObject, KEY_ATTR_TYPE, attrType, opPath,
                          ERR_EXPECTED_ATTR_TYPE) ||
        !fromJSONUnderKey(*opObject, KEY_ATTR_NAME, attrName, opPath,
                          ERR_EXPECTED_ATTR_NAME))
      return failure();

    // Data key must exist
    const json::Value *dataValue = opObject->get(KEY_DATA);
    json::Path dataPath = opPath.field(KEY_DATA);
    if (!dataValue) {
      dataPath.report(ERR_EXPECTED_DATA);
      return failure();
    }

    // Try to decode and set the attribute on the operation
    if (attrType == ATTR_TYPE_BUFFERING) {
      if (failed(setOpAttribute<handshake::ChannelBufPropsAttr>(
              op, attrName, *dataValue, dataPath)))
        return failure();
    }
  }

  return success();
}

LogicalResult
BackAnnotatePass::parseOprdAnnotations(const json::Value &topValue,
                                       json::Path path) {
  // Try to get the "operands" array
  const json::Array *oprdsArray =
      getAnnotationArray(topValue, KEY_OPERANDS, path, ERR_EXPECTED_OPERANDS);
  if (!oprdsArray)
    return failure();

  // Initialize the path
  json::Path arrayPath = path.field(KEY_OPERANDS);

  // Loop over the operand annotations
  for (auto [idx, oprdAnnotation] : llvm::enumerate(*oprdsArray)) {
    json::Path oprdPath = arrayPath.index(idx);

    // Every operand annotation must be an object
    const json::Object *opObject = oprdAnnotation.getAsObject();
    if (!opObject) {
      oprdPath.report(ERR_EXPECTED_OBJECT);
      return failure();
    }

    // Try to find the operation with the referenced name in the IR
    Operation *op = findOperation(*opObject, oprdPath);
    if (!op)
      return failure();

    // Operand annotation must reference an operand index and an attribute
    std::string attrType;
    unsigned oprdIdx;
    if (!fromJSONUnderKey(*opObject, KEY_OPRD_IDX, oprdIdx, oprdPath,
                          ERR_EXPECTED_OPRD_IDX) ||
        !fromJSONUnderKey(*opObject, KEY_ATTR_TYPE, attrType, oprdPath,
                          ERR_EXPECTED_ATTR_TYPE))
      return failure();

    // Check if the operand index makes sense
    if (op->getNumOperands() <= oprdIdx) {
      oprdPath.field(KEY_OPRD_IDX).report(ERR_EXPECTED_OPRD);
      return failure();
    }
    OpOperand &oprd = op->getOpOperand(oprdIdx);

    // Data key must exist
    const json::Value *dataValue = opObject->get(KEY_DATA);
    json::Path dataPath = oprdPath.field(KEY_DATA);
    if (!dataValue) {
      dataPath.report(ERR_EXPECTED_DATA);
      return failure();
    }

    // Try to decode and set the attribute on the operation
    if (attrType == ATTR_TYPE_BUFFERING) {
      if (failed(setOprdAttribute<handshake::ChannelBufPropsAttr>(
              oprd, *dataValue, dataPath)))
        return failure();
    }
  }

  return success();
}

Operation *BackAnnotatePass::findOperation(const json::Object &object,
                                           json::Path path) {
  // Annotation must reference an operation name
  std::string opName;
  if (!fromJSONUnderKey(object, KEY_OPNAME, opName, path, ERR_EXPECTED_OPNAME))
    return nullptr;

  // Try to find the operation with the referenced name in the IR
  Operation *op = getAnalysis<NameAnalysis>().getOp(opName);
  if (!op) {
    path.field(KEY_OPNAME).report(ERR_EXPECTED_OP);
    return nullptr;
  }
  return op;
}

template <typename Attr>
LogicalResult
BackAnnotatePass::setOpAttribute(Operation *op, StringRef attrName,
                                 const json::Value &value, json::Path path) {
  Attr attr;
  if (!fromJSON(value, attr, path, &getContext()))
    return failure();
  op->setAttr(attrName, attr);
  return success();
}

template <typename Attr>
LogicalResult BackAnnotatePass::setOprdAttribute(OpOperand &oprd,
                                                 const json::Value &value,
                                                 json::Path path) {
  Attr attr;
  if (!fromJSON(value, attr, path, &getContext()))
    return failure();
  setOperandAttr(oprd, attr);
  return success();
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createBackAnnotate(const std::string &filepath) {
  return std::make_unique<BackAnnotatePass>(filepath);
}
