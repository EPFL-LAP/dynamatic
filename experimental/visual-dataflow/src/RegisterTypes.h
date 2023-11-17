//===- RegisterTypes.h - GDExtension management -----------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Initialization and termination function for GDExtension library.
//
//===----------------------------------------------------------------------===//

#ifndef VISUAL_DATAFLOW_REGISTER_TYPES_H
#define VISUAL_DATAFLOW_REGISTER_TYPES_H

void initializeModule();
void terminateModule();

#endif // VISUAL_DATAFLOW_REGISTER_TYPES_H
