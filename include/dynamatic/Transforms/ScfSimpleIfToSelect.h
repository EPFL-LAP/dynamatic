//===- ScfSimpleIfToSelect.h - Transform if's into select's -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --scf-simple-if-to-select pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_SCFSIMPLEIFTOSELECT_H
#define DYNAMATIC_TRANSFORMS_SCFSIMPLEIFTOSELECT_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {

std::unique_ptr<dynamatic::DynamaticPass> createScfSimpleIfToSelect();

#define GEN_PASS_DECL_SCFSIMPLEIFTOSELECT
#define GEN_PASS_DEF_SCFSIMPLEIFTOSELECT
#include "dynamatic/Transforms/Passes.h.inc"

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_SCFSIMPLEIFTOSELECT_H
