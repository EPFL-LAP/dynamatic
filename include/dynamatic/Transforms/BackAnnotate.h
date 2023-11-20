//===- BackAnnotate.h - Back-annotate IR from JSON input --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --back-annotate pass, which sets attributes in an
// input IR based on a JSON-formatted back-annotation file. Every attribute one
// wishes to support in the back-annotation pass must be explicitly added to the
// implementation of the pass (open a GitHub issue to request for an attribute
// to be supported).
//
// The JSON file is expected to have the following format.
//
// ```json
//  {
//    "operations": [
//      /* <operation annotation> */
//    ],
//    "operands": [
//      /* <operand annotation> */
//    ]
//  }
// ```
//
// Every "operation annotation" refers to an attribute to set on an MLIR
// operation. Similarly, every "operand annotation" refers to an attribute to
// set (semantically) on an MLIR operand (internally these are stored on the
// operand owning operation since MLIR does not natively support operand
// attributes).
//
// Every annotation must be a JSON object with some required fields (slightly
// different between operations and operands).
//
// ```json
//  {
//    "operations": [
//      {
//        "operation-name": "<name-of-the-op-to-annotate>", /* must exist */
//        "attribute-name": "<name-to-give-to-the-attribute>", /* arbitrary */
//        "attribute-type": "<type-of-the-attribute>", /* must be supported */
//        "attribute-data": <data format specific to the attribute type>
//      },
//      ...
//    ],
//    "operands": [
//      {
//        "operation-name": "<name-of-the-op-to-annotate>", /* must exist */
//        "operand-idx": <integer-index-of-the-operand>, /* must be valid */
//        "attribute-type": "<type-of-the-attribute>", /* must be supported */
//        "attribute-data": <data format specific to the attribute type>
//        /* "attribute-name" automatically determined for operand attributes */
//      },
//      ...
//    ]
//  }
// ```
//
// Note that attribute types may be supported only for operations, only for
// operands, or for both.
//
// At the moment, a single "proof-of-concept" attribute type is supported for
// both operations and operands, it is called "buffering-properties" and maps to
// the `circt::handshake::ChannelBufPropsAttr` attribute internally, which
// encodes buffering constraints for a specific channel (see attribute
// documentation for a description of what each field means). Its expected data
// format is outlined below.
//
// ```json
//  {
//    "operations": [
//      {
//        "operation-name": "some-op-name",
//        "attribute-name": "some-attr-name",
//        "attribute-type": "buffering-properties",
//        "attribute-data": {
//          "minimum-trans": <integer>, /* optional, default 0 */
//          "maximum-trans": <integer>, /* optional, default infinity */
//          "minimum-opaque": <integer>, /* optional, default 0 */
//          "maximum-opaque": <integer>, /* optional, default infinity */
//          "input-delay": <float>, /* optional, default 0.0 */
//          "output-delay": <float>, /* optional, default 0.0 */
//          "unbuf-delay": <float> /* optional, default 0.0 */
//        }
//      }
//    ],
//    "operands": []
//  }
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BACK_ANNOTATE_H
#define DYNAMATIC_TRANSFORMS_BACK_ANNOTATE_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {

#define GEN_PASS_DECL_BACKANNOTATE
#define GEN_PASS_DEF_BACKANNOTATE
#include "dynamatic/Transforms/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass>
createBackAnnotate(const std::string &filepath = "");

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_BACK_ANNOTATE_H