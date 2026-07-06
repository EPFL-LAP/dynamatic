//===- HandshakeSetBufferingProperties.h - Set buf. props. ------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --handshake-set-buffering-properties pass. For now there is
// only a single policy, but it is expected that more will be defined in the
// future. Similarly to the default "fpga20", specifying a new policy amounts to
// writing a simple function that will be called on each channel present in the
// dataflow circuit and modify, if necessary, the buffering properties
// associated with it. The logic for fetching and writing back that data to the
// IR is conveniently hidden to reduce the amount of boilerplate code and
// improve performance.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Transforms/BufferPlacement/Utils/BufferingSupport.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"

using namespace dynamatic;
using namespace dynamatic::buffer;
// [START Boilerplate code for the MLIR pass]

#include "dynamatic/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
#define GEN_PASS_DEF_HANDSHAKESETBUFFERINGPROPERTIES
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

/// End of error message when there is a conflict between the constraint-setting
/// logic and already existing buffering constraints.
static const llvm::StringLiteral
    ERR_CONFLICT("but previous channel constraints prevent buffering. "
                 "Resulting circuit may deadlock.");

/// Makes the channel unbufferizable.
static void makeUnbufferizable(Value val) {
  assert(!val.use_empty() &&
         "Cannot treat a value without a use as a channel!");

  Channel channel(val, true);
  channel.props->maxOpaque = 0;
  channel.props->maxTrans = 0;
}

// The speculator requires specific buffers
// for correctness
static LogicalResult
setSpeculatorBufferingProperties(handshake::FuncOp funcOp) {
  auto specOps = funcOp.getOps<handshake::SpeculatorOp>();
  auto specCount = std::distance(specOps.begin(), specOps.end());
  if (specCount > 1) {
    funcOp.emitError() << "Expected at most one SpeculatorOp";
    return failure();
  }
  // no spec op = successful buffer placement
  if (specCount == 0)
    return success();

  auto specOp = *specOps.begin();

  // dataOut is volatile
  // (value can change without being accepted,
  // from a predicted value
  // to re-sending an mis-predicted value)
  // and so is not safe to connect to an eager fork.
  // We place a buffer here for safety
  Value dataOut = specOp.getDataOut();
  Channel dataChannel(dataOut, true);
  dataChannel.props->minTrans = std::max(dataChannel.props->minTrans, 1U);

  // issueCtrl is volatile
  // (value can change without being accepted,
  // from spec to resend)
  // and so if not safe to connect to an eager fork
  // We place a buffer here for safety.
  Value issueCtrl = specOp.getIssueCtrl();
  Channel issueChannel(issueCtrl, true);
  issueChannel.props->minTrans = std::max(issueChannel.props->minTrans, 1U);

  // historyCtrl is on a path from a lazy fork
  // to a join
  // and so needs a buffer to prevent deadlock
  Value historyCtrl = specOp.getHistoryCtrl();
  Channel resolveChannel(historyCtrl, true);
  resolveChannel.props->minTrans = std::max(resolveChannel.props->minTrans, 1U);

  // The speculator's KILL_ONLY_DATA state stalls the data input for
  // 1 cycle during misspeculation recovery. A transparent buffer on
  // the data input absorbs this stall and prevents it from propagating
  // upstream and causing throughput loss.
  Value dataIn = specOp.getDataIn();
  Channel dataInChannel(dataIn, true);
  dataInChannel.props->minTrans = std::max(dataInChannel.props->minTrans, 1U);

  return success();
}

/// Sets buffering constraints related to the LSQ's control path. Output
/// channels of group-allocation-signal-defining (lazy-)forks to the LSQ must be
/// buffered in a particular way:
/// - direct outputs between the fork and LSQs must remain unbuffered
/// - outputs going to next group allocations to the same LSQ must have their
/// data/valid paths cut
/// - all other outputs must have their ready path cut
static void setLSQControlConstraints(handshake::LSQOp lsqOp) {
  LSQPorts ports = lsqOp.getPorts();
  ValueRange lsqInputs = lsqOp.getOperands();

  for (LSQGroup &group : ports.getGroups()) {
    // Control signal must come from a fork for this constraint to apply
    Value ctrlVal = lsqInputs[group->ctrlPort->getCtrlInputIndex()];
    Operation *ctrlDefOp = ctrlVal.getDefiningOp();
    if (!mlir::isa_and_present<handshake::ForkOp, handshake::LazyForkOp>(
            ctrlDefOp))
      continue;

    // Force placement of an opaque buffer slot on other fork output channels
    // triggering group allocations to the same LSQ. Other output channels not
    // part of the control paths to the LSQ get a transparent buffer slot
    SmallVector<Value> ctrlPaths = lsqOp.getControlPaths(ctrlDefOp);
    llvm::SmallSetVector<Value, 4> ctrlPathSet(ctrlPaths.begin(),
                                               ctrlPaths.end());
    for (OpResult forkRes : ctrlDefOp->getResults()) {
      // Channels connecting directly to LSQs should be left alone (group
      // allocation signals have already been rendered unbufferizable before,
      // i.e., in setFPGA20Properties)
      if (isa<handshake::LSQOp>(*forkRes.getUsers().begin()))
        continue;

      if (ctrlPathSet.contains(forkRes)) {
        // Path goes to other group allocation to the same LSQ
        Channel channel(forkRes, true);
        if (channel.props->maxOpaque.value_or(1) > 0) {
          channel.props->minOpaque = std::max(channel.props->minOpaque, 1U);
        } else {
          OpOperand &oprd = channel.getOperand();
          ctrlDefOp->emitWarning()
              << "Fork result " << forkRes.getResultNumber() << " ("
              << getUniqueName(oprd)
              << ") is on path to other LSQ group allocation and should "
                 "have its data/valid paths cut, "
              << ERR_CONFLICT;
        }
      } else {
        // Path does not go to the same LSQ
        Channel channel(forkRes, true);
        if (channel.props->maxTrans.value_or(1) > 0) {
          channel.props->minTrans = std::max(channel.props->minTrans, 1U);
        } else {
          OpOperand &oprd = channel.getOperand();
          ctrlDefOp->emitWarning()
              << "Fork result " << forkRes.getResultNumber() << " ("
              << getUniqueName(oprd)
              << ") is *not* on path to other LSQ group allocation and "
                 "should have its ready path cut, "
              << ERR_CONFLICT;
        }
      }
    }
  }
}

static LogicalResult setFPGA20Properties(handshake::FuncOp funcOp) {
  // See docs/Specs/Buffering.md
  // A merge with more than one input should have at least one
  // buffer slot at its output, and this is necessary only if
  // the merge is on a cycle.
  for (handshake::MergeOp mergeOp : funcOp.getOps<handshake::MergeOp>()) {
    if (mergeOp->getNumOperands() > 1) {
      for (OpResult mergeRes : mergeOp->getResults()) {
        Channel channel(mergeRes, true);
        if (isChannelOnCycle(mergeRes)) {
          channel.props->minSlots = std::max(channel.props->minSlots, 1U);
        }
      }
    }
  }

  // See docs/Specs/Buffering.md
  // To mitigate the latency asymmetry between LSQ group allocation
  // and the Store/Load operations, we set a minimum number of buffer
  // slots at Store/Load's input.
  // This is a temporary workaround and a better solution is needed.
  for (handshake::StoreOp storeOp : funcOp.getOps<handshake::StoreOp>()) {
    auto memOp = findMemInterface(storeOp.getAddressResult());
    if (!mlir::isa_and_present<handshake::LSQOp>(memOp))
      continue;

    for (Value operand : storeOp->getOperands()) {
      Channel channel(operand, true);
      Operation *defOp = operand.getDefiningOp();

      if (defOp) {
        channel.props->minSlots = std::max(channel.props->minSlots, 1U);
      }
    }
  }

  for (handshake::LoadOp loadOp : funcOp.getOps<handshake::LoadOp>()) {
    auto memOp = findMemInterface(loadOp.getAddressResult());
    if (!mlir::isa_and_present<handshake::LSQOp>(memOp))
      continue;

    for (Value operand : loadOp->getOperands()) {
      Channel channel(operand, true);
      Operation *defOp = operand.getDefiningOp();

      if (defOp &&
          !isa<handshake::MemoryOpInterface, handshake::ConstantOp>(defOp)) {
        channel.props->minSlots = std::max(channel.props->minSlots, 1U);
      }
    }
  }

  // See docs/Specs/Buffering.md
  // Memrefs are not real edges in the graph and are therefore unbufferizable
  for (BlockArgument arg : funcOp.getArguments()) {
    makeUnbufferizable(arg);
  }

  // Ports of memory interfaces are unbufferizable
  for (auto memOp : funcOp.getOps<handshake::MemoryOpInterface>()) {
    FuncMemoryPorts ports = getMemoryPorts(memOp);
    for (size_t i = 0, e = ports.getNumGroups(); i < e; ++i) {
      for (Value inputVal : ports.getGroupInputs(i))
        makeUnbufferizable(inputVal);
      for (Value outputVal : ports.getGroupResults(i))
        makeUnbufferizable(outputVal);
    }
    for (Value inputVal : ports.getInterfacesInputs())
      makeUnbufferizable(inputVal);
    for (Value outputVal : ports.getInterfacesResults())
      makeUnbufferizable(outputVal);
  }

  // See docs/Specs/Buffering.md
  // Control paths to LSQs have specific properties
  for (handshake::LSQOp lsqOp : funcOp.getOps<handshake::LSQOp>())
    setLSQControlConstraints(lsqOp);

  if (failed(setSpeculatorBufferingProperties(funcOp)))
    return failure();

  return success();
}

namespace {
/// Simple pass driver that runs a specific buffering properties setting
/// policy on each Handshake function in the IR.
struct HandshakeSetBufferingPropertiesPass
    : public dynamatic::impl::HandshakeSetBufferingPropertiesBase<
          HandshakeSetBufferingPropertiesPass> {

  using HandshakeSetBufferingPropertiesBase::
      HandshakeSetBufferingPropertiesBase;

  void runDynamaticPass() override {
    mlir::ModuleOp modOp = getOperation();
    // Check that the provided version is valid
    if (version != "fpga20") {
      modOp->emitError() << "Unkwown version \"" << version
                         << "\", expected one of [fpga20]";
      return signalPassFailure();
    }

    // Add properties to channels inside each function
    for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
      // Buffer placement requires that all values are used exactly once
      if (failed(verifyIRMaterialized(funcOp))) {
        funcOp.emitOpError() << ERR_NON_MATERIALIZED_FUNC;
        return signalPassFailure();
      }
      if (failed(setFPGA20Properties(funcOp)))
        return signalPassFailure();
    }
  };
};

} // namespace
