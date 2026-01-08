//===----------------------------------------------------------------------===//
// All the files related to the description of the Synth dialect
// have been modified from the original version present in the
// circt project at the following link:
// https://github.com/llvm/circt/tree/main/
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Dialect/Synth/SynthDialect.h"
#include "dynamatic/Dialect/HW/HWOps.h"
#include "dynamatic/Dialect/Synth/SynthOps.h"

using namespace dynamatic;
using namespace synth;

void SynthDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "dynamatic/Dialect/Synth/Synth.cpp.inc"
      >();
}

Operation *SynthDialect::materializeConstant(OpBuilder &builder,
                                             Attribute value, Type type,
                                             Location loc) {
  // Integer constants.
  if (auto intType = dyn_cast<IntegerType>(type))
    if (auto attrValue = dyn_cast<IntegerAttr>(value))
      return builder.create<hw::ConstantOp>(loc, type, attrValue);
  // hw::ConstantOp::create(builder, loc, type, attrValue);
  return nullptr;
}

#include "dynamatic/Dialect/Synth/SynthDialect.cpp.inc"
