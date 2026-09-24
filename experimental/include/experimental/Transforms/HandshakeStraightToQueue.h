

#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKE_STRAIGHT_TO_QUEUE_H
#define DYNAMATIC_TRANSFORMS_HANDSHAKE_STRAIGHT_TO_QUEUE_H

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "experimental/Support/FtdSupport.h"

namespace dynamatic {
namespace experimental {

/// Per-block edge information captured from the multi-block CFG before
/// flattening. Used to reconstruct a ShadowCFG afterwards.
struct CapturedEdgeInfo {
  bool isConditional = false;
  bool hasSuccessors = false;
  unsigned trueSuccIdx = 0;
  unsigned falseSuccIdx = 0;
  unsigned uncondSuccIdx = 0;
};

void captureCFGTopology(handshake::FuncOp funcOp, unsigned &numBlocks,
                        SmallVector<CapturedEdgeInfo> &edges,
                        DenseMap<unsigned, Value> &capturedConditions);

ftd::ShadowCFG buildShadowFromCapturedTopology(
    OpBuilder &builder, handshake::FuncOp funcOp, unsigned numBlocks,
    const SmallVector<CapturedEdgeInfo> &edges,
    const DenseMap<unsigned, Value> &capturedConditions);

} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKE_STRAIGHT_TO_QUEUE_H
