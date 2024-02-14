//===- HWTypeInterfaces.h - Declare HW type interfaces ----------*- C++ -*-===//
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
// This file declares type interfaces for the HW Dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_DIALECT_HW_HW_TYPE_INTERFACES_H
#define DYNAMATIC_DIALECT_HW_HW_TYPE_INTERFACES_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Types.h"

namespace dynamatic {
namespace hw {
namespace FieldIdImpl {
uint64_t getMaxFieldID(Type);

std::pair<::mlir::Type, uint64_t> getSubTypeByFieldID(Type, uint64_t fieldID);

::mlir::Type getFinalTypeByFieldID(Type type, uint64_t fieldID);

std::pair<uint64_t, bool> projectToChildFieldID(Type, uint64_t fieldID,
                                                uint64_t index);

std::pair<uint64_t, uint64_t> getIndexAndSubfieldID(Type type,
                                                    uint64_t fieldID);

uint64_t getFieldID(Type type, uint64_t index);

uint64_t getIndexForFieldID(Type type, uint64_t fieldID);

} // namespace FieldIdImpl
} // namespace hw
} // namespace dynamatic

#include "dynamatic/Dialect/HW/HWTypeInterfaces.h.inc"

#endif // DYNAMATIC_DIALECT_HW_HW_TYPE_INTERFACES_H
