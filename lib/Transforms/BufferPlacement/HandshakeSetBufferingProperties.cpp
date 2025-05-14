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

#include "dynamatic/Transforms/BufferPlacement/HandshakeSetBufferingProperties.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"

using namespace dynamatic;
using namespace dynamatic::buffer;

/// End of error message when there is a conflict between the constraint-setting
/// logic and already existing buffering constraints.
static const llvm::StringLiteral
    ERR_CONFLICT("but previous channel constraints prevent buffering. "
                 "Resulting circuit may deadlock.");

/// Makes the channel unbufferizable.
static void makeUnbufferizable(Value val) {
  Channel channel(val, true);
  channel.props->maxOpaque = 0;
  channel.props->maxTrans = 0;
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

void dynamatic::buffer::setFPGA20Properties(handshake::FuncOp funcOp) {
  // Merges with more than one input should have at least a transparent slot
  // at their output
  for (handshake::MergeOp mergeOp : funcOp.getOps<handshake::MergeOp>()) {
    if (mergeOp->getNumOperands() > 1) {
      Channel channel(mergeOp.getResult(), true);
      channel.props->minTrans = std::max(channel.props->minTrans, 1U);
    }
  }
  
  for (handshake::StoreOp storeOp : funcOp.getOps<handshake::StoreOp>()) {
    bool connectedToLSQ = false;
    for (Operation *user : storeOp->getUsers()) {
      if (isa<handshake::LSQOp>(user)) {
        connectedToLSQ = true;
        break;
      }
    }
  
    if (!connectedToLSQ)
      continue;
  
    for (Value operand : storeOp->getOperands()) {
      Channel channel(operand, true);
      if (channel.props->maxTrans.value_or(1) > 0) {
        channel.props->minTrans = std::max(channel.props->minTrans, 1U);
      } else {
        storeOp->emitWarning()
            << "Store input channel (connected to LSQ) should have transparent buffer, but not allowed";
      }
    }
  }

  for (handshake::LoadOp loadOp : funcOp.getOps<handshake::LoadOp>()) {
    bool connectedToLSQ = false;
    for (Operation *user : loadOp->getUsers()) {
      if (isa<handshake::LSQOp>(user)) {
        connectedToLSQ = true;
        break;
      }
    }

    if (!connectedToLSQ)
      continue;

    for (Value operand : loadOp->getOperands()) {
      Channel channel(operand, true);
      if (channel.props->maxTrans.value_or(1) > 0) {
        channel.props->minTrans = std::max(channel.props->minTrans, 1U);
      } else {
        loadOp->emitWarning()
            << "Load input channel (connected to LSQ) should have transparent buffer, but not allowed";
      }
    }
  }

  // // Build the CFG adjacency: succ / pred maps
  // DenseMap<Operation*, SmallVector<Operation*, 8>> succ, pred;
  // funcOp.walk([&](Operation *op) {
  //   for (Value result : op->getResults()) {
  //     for (Operation *user : result.getUsers()) {
  //       succ[op].push_back(user);
  //       pred[user].push_back(op);
  //     }
  //   }
  // });

  // // Helper: BFS find any path src→dst, optionally avoiding 'blocked'
  // auto findPath = [&](Operation *src,
  //                     Operation *dst,
  //                     SmallVectorImpl<Operation*> &outPath,
  //                     const SmallPtrSetImpl<Operation*> *blocked = nullptr) {
  //   DenseMap<Operation*, Operation*> parent;
  //   SmallVector<Operation*, 64> queue;
  //   queue.push_back(src);
  //   parent[src] = nullptr;

  //   bool found = false;
  //   for (size_t i = 0; i < queue.size() && !found; ++i) {
  //     Operation *cur = queue[i];
  //     for (Operation *nxt : succ[cur]) {
  //       if (parent.count(nxt))
  //         continue;
  //       if (blocked && blocked->contains(nxt))
  //         continue;
  //       parent[nxt] = cur;
  //       if (nxt == dst) {
  //         found = true;
  //         break;
  //       }
  //       queue.push_back(nxt);
  //     }
  //   }
  //   if (!found)
  //     return false;
  //   // reconstruct path
  //   for (Operation *it = dst; it != nullptr; it = parent[it])
  //     outPath.push_back(it);
  //   std::reverse(outPath.begin(), outPath.end());
  //   return true;
  // };

  // // For each StoreOp that feeds directly into an LSQOp, apply the new logic
  // for (handshake::StoreOp storeOp : funcOp.getOps<handshake::StoreOp>()) {
  //   // (a) find the adjacent LSQOp, if any
  //   handshake::LSQOp lsqOp = nullptr;
  //   for (OpResult res : storeOp.getResults()) {
  //     for (Operation *user : res.getUsers()) {
  //       if ((lsqOp = dyn_cast<handshake::LSQOp>(user)))
  //         break;
  //     }
  //     if (lsqOp) break;
  //   }
  //   // If no direct store→lsq adjacency
  //   if (!lsqOp)
  //     continue;

  //   // (b) backwards BFS from storeOp to collect all reachable ForkOps
  //   SmallVector<Operation*, 64> work{storeOp};
  //   DenseSet<Operation*> visited;
  //   SmallPtrSet<Operation*, 16> forkSet;
  //   while (!work.empty()) {
  //     Operation *cur = work.pop_back_val();
  //     if (!visited.insert(cur).second)
  //       continue;
  //     if (auto forkOp = dyn_cast<handshake::ForkOp>(cur))
  //       forkSet.insert(forkOp);
  //     for (Operation *p : pred[cur])
  //       work.push_back(p);
  //   }

  //   // (c) for each fork, try to find two disjoint paths:
  //   // (1) fork→…→store, (2) fork→…→lsq, disjoint except at fork/lsq
  //   for (Operation *forkOp : forkSet) {
  //     // path from fork→store
  //     SmallVector<Operation*, 32> pathFS;
  //     if (!findPath(forkOp, storeOp, pathFS))
  //       continue;

  //     // build block set from that path (excluding forkOp and lsqOp)
  //     SmallPtrSet<Operation*, 32> blocked(pathFS.begin(), pathFS.end());
  //     blocked.erase(forkOp);
  //     blocked.erase(lsqOp.getOperation());

  //     // try to find fork→lsq avoiding blocked nodes
  //     SmallVector<Operation*, 32> pathFL;
  //     if (findPath(forkOp, lsqOp.getOperation(), pathFL, &blocked)) {
  //       // success: set minTrans on each input channel of this storeOp
  //       for (Value operand : storeOp.getOperands()) {
  //         Channel channel(operand, true);
  //         if (channel.props->maxTrans.value_or(1) > 0) {
  //           channel.props->minTrans = std::max(channel.props->minTrans, 1U);
  //         } else {
  //           storeOp->emitWarning()
  //             << "Store input channel (connected to LSQ) should have "
  //             << "transparent buffer, but not allowed";
  //         }
  //       }
  //       // no need to test other forks for this storeOp
  //       break;
  //     }
  //   }
  // }
  
  // Memrefs are not real edges in the graph and are therefore unbufferizable
  for (BlockArgument arg : funcOp.getArguments())
    makeUnbufferizable(arg);

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

  // Control paths to LSQs have specific properties
  for (handshake::LSQOp lsqOp : funcOp.getOps<handshake::LSQOp>())
    setLSQControlConstraints(lsqOp);
}

namespace {
/// Simple pass driver that runs a specific buffering properties setting
/// policy on each Handshake function in the IR.
struct HandshakeSetBufferingPropertiesPass
    : public dynamatic::buffer::impl::HandshakeSetBufferingPropertiesBase<
          HandshakeSetBufferingPropertiesPass> {

  HandshakeSetBufferingPropertiesPass(const std::string &version) {
    this->version = version;
  }

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
      setFPGA20Properties(funcOp);
    }
  };
};

} // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::buffer::createHandshakeSetBufferingProperties(
    const std::string &version) {
  return std::make_unique<HandshakeSetBufferingPropertiesPass>(version);
}
