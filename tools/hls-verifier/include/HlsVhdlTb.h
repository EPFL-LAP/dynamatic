//===- HlsVhdlTb.h ----------------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HLS_VERIFIER_HLS_VHDL_TB_H
#define HLS_VERIFIER_HLS_VHDL_TB_H

#include "VerificationContext.h"
#include "mlir/Support/IndentedOstream.h"
#include <string>
#include <vector>

using namespace std;

void vhdlTbCodegen(VerificationContext &ctx);

#endif // HLS_VERIFIER_HLS_VHDL_TB_H
