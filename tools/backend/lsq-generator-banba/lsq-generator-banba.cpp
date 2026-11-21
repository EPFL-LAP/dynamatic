#include "circt/Dialect/HW/HWDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"

int main() {
  // Loading a CIRCT dialect makes this program depend on CIRCT's headers and
  // libraries, which checks that Dynamatic's build finds both
  mlir::MLIRContext context;
  context.loadDialect<circt::hw::HWDialect>();

  llvm::outs() << "hello world\n";
  return 0;
}
