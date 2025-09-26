#include "SpecV1Adaptor.h"
#include "JSONImporter.h"
#include "SpecV2Lib.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::speculationv2;

namespace dynamatic {
namespace experimental {
namespace speculationv2 {

// Implement the base class and auto-generated create functions.
// Must be called from the .cpp file to avoid multiple definitions
#define GEN_PASS_DEF_SPECV1ADAPTOR
#include "experimental/Transforms/Passes.h.inc"

} // namespace speculationv2
} // namespace experimental
} // namespace dynamatic

struct SpecV1AdaptorPass
    : public dynamatic::experimental::speculationv2::impl::SpecV1AdaptorBase<
          SpecV1AdaptorPass> {
  using SpecV1AdaptorBase<SpecV1AdaptorPass>::SpecV1AdaptorBase;
  void runDynamaticPass() override;
};

void SpecV1AdaptorPass::runDynamaticPass() {
  ModuleOp modOp = getOperation();

  // Support only one funcOp
  assert(std::distance(modOp.getOps<FuncOp>().begin(),
                       modOp.getOps<FuncOp>().end()) == 1 &&
         "Expected a single FuncOp in the module");

  FuncOp funcOp = *modOp.getOps<FuncOp>().begin();

  introduceGSAMux(funcOp, branchBB);

  OpBuilder builder(funcOp.getContext());
  for (auto branch : funcOp.getOps<ConditionalBranchOp>()) {
    auto brBB = getLogicBB(branch);
    if (!brBB || *brBB != branchBB)
      continue;

    branch->setAttr("specv1_adaptor_inner_loop", builder.getBoolAttr(true));
  }
  for (auto cmerge : funcOp.getOps<ControlMergeOp>()) {
    auto cmBB = getLogicBB(cmerge);
    if (!cmBB || *cmBB != mergeBB)
      continue;

    cmerge->setAttr("specv1_adaptor_inner_loop", builder.getBoolAttr(true));
  }
  for (auto mux : funcOp.getOps<MuxOp>()) {
    auto muxBB = getLogicBB(mux);
    if (!muxBB || *muxBB != mergeBB)
      continue;

    mux->setAttr("specv1_adaptor_inner_loop", builder.getBoolAttr(true));
  }

  SmallVector<unsigned> loopBBs;
  for (unsigned bb = branchBB; bb <= mergeBB; bb++) {
    loopBBs.push_back(bb);
  }
  DenseMap<unsigned, unsigned> bbMap = unifyBBs(loopBBs, funcOp);
  if (bbMapping != "") {
    // Convert bbMap to a json file
    std::ofstream csvFile(bbMapping);
    csvFile << "before,after\n";
    for (const auto &entry : bbMap) {
      csvFile << entry.first << "," << entry.second << "\n";
    }
    csvFile.close();
  }

  recalculateMCBlocks(funcOp);
}
