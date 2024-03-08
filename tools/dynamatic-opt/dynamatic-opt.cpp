//===- dynamatic-opt.cpp - The dynamatic-opt driver -------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the dynamatic-opt tool, which is the Dynamatic analog of
// mlir-opt. It allows access to all compiler passes that Dynamatic users may
// care about, that is all passes defined in the superproject as well as certain
// passes defined within upstream MLIR.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/InitAllDialects.h"
#include "dynamatic/InitAllPasses.h"
#include "experimental/InitAllPasses.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "tutorials/CreatingPasses/InitAllPasses.h"

// Defined in the test directory, no public header.
namespace dynamatic {
namespace experimental {
namespace test {
void registerTestCDGAnalysisPass();
} // namespace test
} // namespace experimental
} // namespace dynamatic

void registerTestPasses() {
  dynamatic::experimental::test::registerTestCDGAnalysisPass();
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register standard MLIR passes we care about
  mlir::registerTransformsPasses();
  mlir::affine::registerAffinePasses();
  mlir::arith::registerArithPasses();
  mlir::func::registerFuncPasses();
  mlir::memref::registerMemRefPasses();
  mlir::registerSCFPasses();

  // Register specific dialects and passes we want
  dynamatic::registerAllDialects(registry);
  dynamatic::registerAllPasses();
  dynamatic::tutorials::registerAllPasses();
  dynamatic::experimental::registerAllPasses();
  registerTestPasses();

  // Register the standard passes we want
  mlir::registerCSEPass();
  mlir::registerSCCPPass();
  mlir::registerInlinerPass();
  mlir::registerCanonicalizerPass();
  mlir::registerSymbolDCEPass();

  return mlir::failed(mlir::MlirOptMain(
      argc, argv, "Dynamatic modular optimizer driver", registry));
}
