#include "dynamatic/Transforms/HandshakeSetTimingDBPath.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/ResourceBlobManager.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

namespace dynamatic {

// include tblgen base class definition
#define GEN_PASS_DEF_HANDSHAKESETTIMINGDBPATH_H
#include "dynamatic/Transforms/Passes.h.inc"

} // namespace dynamatic

namespace {

struct HandshakeSetTimingDBPathPass
    : public dynamatic::impl::HHandshakeSetTimingDBPathBase<
          HandshakeSetTimingDBPathPass> {
public:
  // use tblgen constructors from base class
  using HandshakeMarkFPUImplBase::HandshakeMarkFPUImplBase;

  // inherited TableGen Pass Options:
  // std::string jsonPath

  void runDynamaticPass() override;
};


void HandshakeSetTimingDBPathPass::runDynamaticPass() {
    if (this->jsonPath.empty()) {
      getOperation()->emitError("Missing --json-path");
      return signalPassFailure();
    }

    std::ifstream file(this->jsonPath);
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    getContext()->getBlobResourceManager().addResource(
      "timingDB",
      getContext()->getOrLoadDialect("handshake"),
      content
    );
  }
};