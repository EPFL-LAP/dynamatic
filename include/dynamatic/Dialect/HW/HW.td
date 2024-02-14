//===- HW.td - HW dialect definition -----------------------*- tablegen -*-===//
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
// This is the top level file for the HW dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_DIALECT_HW_HW_TD
#define DYNAMATIC_DIALECT_HW_HW_TD

include "dynamatic/Dialect/HW/HWDialect.td"
include "dynamatic/Dialect/HW/HWAttributes.td"
include "dynamatic/Dialect/HW/HWAttributesNaming.td"

include "dynamatic/Dialect/HW/HWTypesImpl.td"
include "dynamatic/Dialect/HW/HWTypes.td"

include "dynamatic/Dialect/HW/HWOpInterfaces.td"
include "dynamatic/Dialect/HW/HWTypeInterfaces.td"
include "dynamatic/Dialect/HW/HWMiscOps.td"
include "dynamatic/Dialect/HW/HWAggregates.td"
include "dynamatic/Dialect/HW/HWStructure.td"
include "dynamatic/Dialect/HW/HWTypeDecls.td"

#endif // DYNAMATIC_DIALECT_HW_HW_TD
