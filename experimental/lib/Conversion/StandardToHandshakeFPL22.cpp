// TODO

#include "experimental/Conversion/StandardToHandshakeFPL22.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;
using namespace circt;

namespace {

struct StandardToHandshakeFPL22Pass
    : public dynamatic::experimental::impl::StandardToHandshakeFPL22Base<
          StandardToHandshakeFPL22Pass> {

  void runOnOperation() override { llvm::outs() << "My pass is running!\n"; };
};
} // namespace

namespace dynamatic {
namespace experimental {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createStandardToHandshakeFPL22Pass() {
  return std::make_unique<StandardToHandshakeFPL22Pass>();
}

} // namespace experimental
} // namespace dynamatic