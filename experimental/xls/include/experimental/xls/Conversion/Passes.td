//===- Passes.td - XLS conversion passes definition -------*- tablegen -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definition of all XLS conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef XLS_CONVERSION_PASSES_TD
#define XLS_CONVERSION_PASSES_TD

include "dynamatic/Support/Passes.td"
include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// HandshakeToXls
//===----------------------------------------------------------------------===//

def HandshakeToXls : DynamaticPass<"lower-handshake-to-xls",
                                  ["mlir::xls::XlsDialect"]> {
  let summary = "Lowers Handshake to XLS.";
  let description = [{
    Lowers Handshake IR into a network of XLS procs connected by channels.
  }];
  let constructor = "dynamatic::experimental::xls::createHandshakeToXlsPass()";
}

#endif // XLS_CONVERSION_PASSES_TD

