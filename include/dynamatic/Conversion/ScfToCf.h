//===- ScfToCf.h - Lower scf ops to unstructured control flow ---*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --lower-scf-to-cf pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_SCFTOCF_H
#define DYNAMATIC_TRANSFORMS_SCFTOCF_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {

#define GEN_PASS_DECL_SCFTOCF
#define GEN_PASS_DEF_SCFTOCF
#include "dynamatic/Conversion/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass> createLowerScfToCf();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_SCFTOCF_H
