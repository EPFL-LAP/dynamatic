//===- DOTReformat.h - Reformats a .dot file --------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The DOTReformat class is a temporary fix to the dot file being formatted wr-
// onfully so it can be parsed correctly.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/LogicalResult.h"
#include <string>

mlir::LogicalResult reformatDot(const std::string &inputFileName,
                                const std::string &outputFileName);
