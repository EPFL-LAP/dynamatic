//===- BlifImporterPass.h - Import BLIF files --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the BlifImporterPass, which imports a BLIF file and
// generates a corresponding Synth circuit.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BLIFIMPORTER_H
#define DYNAMATIC_TRANSFORMS_BLIFIMPORTER_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/BlifImporter/BlifImporterSupport.h"
#include "mlir/IR/DialectRegistry.h"

namespace dynamatic {

#define GEN_PASS_DECL_BLIFIMPORTER
#define GEN_PASS_DEF_BLIFIMPORTER
#include "dynamatic/Transforms/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass> createBlifImporterPass();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_BLIFIMPORTER_H
