//===- BlifExporterPass.h - Export BLIF files --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the BlifExporterPass, which exports a Synth circuit to a
// BLIF file.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BLIFEXPORTER_H
#define DYNAMATIC_TRANSFORMS_BLIFEXPORTER_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/BlifExporter/BlifExporterSupport.h"
#include "mlir/IR/DialectRegistry.h"

namespace dynamatic {

#define GEN_PASS_DECL_BLIFEXPORTER
#define GEN_PASS_DEF_BLIFEXPORTER
#include "dynamatic/Transforms/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass> createBlifExporterPass();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_BLIFEXPORTER_H
