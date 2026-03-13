//===- BlifImporterSupport.h - Support functions for BLIF importer -*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the strings used to represent the different structures in
// a blif file.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

// Constant string to represent .name structure in the blif file
static llvm::StringLiteral LIT_NAMES(".names");
// Constant string to represent .latch structure in the blif file
static llvm::StringLiteral LIT_LATCH(".latch");
// Constant string to represent .inputs structure in the blif file
static llvm::StringLiteral LIT_INPUTS(".inputs");
// Constant string to represent .outputs structure in the blif file
static llvm::StringLiteral LIT_OUTPUTS(".outputs");
// Constant string to represent .model structure in the blif file
static llvm::StringLiteral LIT_MODEL(".model");
// Constant string to represent .subckt structure in the blif file
static llvm::StringLiteral LIT_SUBCKT(".subckt");
// Constant string to represent the end of the module in the blif file
static llvm::StringLiteral LIT_END(".end");