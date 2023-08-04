//===- dynamatic-opt.cpp - The dynamatic-opt driver -------------*- C++ -*-===//
//
// This file implements the dynamatic-opt tool, which is the Dynamatic analog of
// mlir-opt. It allows access to all compiler passes that Dynamatic users may
// care about, that is all passes defined in the superproject as well as certain
// passes defined within upstream MLIR, Polygeist, and CIRCT.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/InitAllDialects.h"
#include "dynamatic/InitAllPasses.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "tutorials/InitAllPasses.h"
//#include "experimental/InitAllPasses.h" // TODO: fix linking error

// Defined in the test directory, no public header.
namespace experimental {
namespace test {
void registerTestCDGAnalysisPass();
} // namespace test
} // namespace experimental

void registerTestPasses() { experimental::test::registerTestCDGAnalysisPass(); }

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register specific dialects and passes we want
  dynamatic::registerAllDialects(registry);
  dynamatic::registerAllPasses();
  dynamatic::tutorials::registerAllPasses();
  registerTestPasses();
  //experimental::registerAllExpPasses(); // TODO: fix linking error

  // Register the standard passes we want
  mlir::registerCSEPass();
  mlir::registerSCCPPass();
  mlir::registerInlinerPass();
  mlir::registerCanonicalizerPass();
  mlir::registerSymbolDCEPass();

  return mlir::failed(mlir::MlirOptMain(
      argc, argv, "Dynamatic modular optimizer driver", registry));
}
