//===- HWTypeInterfaces.cpp - Implement HW type interfaces ------*- C++ -*-===//
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
// This file implements type interfaces of the HW Dialect.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Dialect/HW/HWTypeInterfaces.h"

using namespace mlir;
using namespace dynamatic;
using namespace hw;
using namespace FieldIdImpl;

Type dynamatic::hw::FieldIdImpl::getFinalTypeByFieldID(Type type,
                                                       uint64_t fieldID) {
  std::pair<Type, uint64_t> pair(type, fieldID);
  while (pair.second) {
    if (auto ftype = dyn_cast<FieldIDTypeInterface>(pair.first))
      pair = ftype.getSubTypeByFieldID(pair.second);
    else
      llvm::report_fatal_error("fieldID indexing into a non-aggregate type");
  }
  return pair.first;
}

std::pair<Type, uint64_t>
dynamatic::hw::FieldIdImpl::getSubTypeByFieldID(Type type, uint64_t fieldID) {
  if (!fieldID)
    return {type, 0};
  if (auto ftype = dyn_cast<FieldIDTypeInterface>(type))
    return ftype.getSubTypeByFieldID(fieldID);

  llvm::report_fatal_error("fieldID indexing into a non-aggregate type");
}

uint64_t dynamatic::hw::FieldIdImpl::getMaxFieldID(Type type) {
  if (auto ftype = dyn_cast<FieldIDTypeInterface>(type))
    return ftype.getMaxFieldID();
  return 0;
}

std::pair<uint64_t, bool>
dynamatic::hw::FieldIdImpl::projectToChildFieldID(Type type, uint64_t fieldID,
                                                  uint64_t index) {
  if (auto ftype = dyn_cast<FieldIDTypeInterface>(type))
    return ftype.projectToChildFieldID(fieldID, index);
  return {0, fieldID == 0};
}

uint64_t dynamatic::hw::FieldIdImpl::getIndexForFieldID(Type type,
                                                        uint64_t fieldID) {
  if (auto ftype = dyn_cast<FieldIDTypeInterface>(type))
    return ftype.getIndexForFieldID(fieldID);
  return 0;
}

uint64_t dynamatic::hw::FieldIdImpl::getFieldID(Type type, uint64_t fieldID) {
  if (auto ftype = dyn_cast<FieldIDTypeInterface>(type))
    return ftype.getFieldID(fieldID);
  return 0;
}

std::pair<uint64_t, uint64_t>
dynamatic::hw::FieldIdImpl::getIndexAndSubfieldID(Type type, uint64_t fieldID) {
  if (auto ftype = dyn_cast<FieldIDTypeInterface>(type))
    return ftype.getIndexAndSubfieldID(fieldID);
  return {0, fieldID == 0};
}

#include "dynamatic/Dialect/HW/HWTypeInterfaces.cpp.inc"
