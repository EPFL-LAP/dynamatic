#include "dynamatic/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  dynamatic::registerAllDialects(registry);
  return failed(mlir::MlirLspServerMain(argc, argv, registry));
}
