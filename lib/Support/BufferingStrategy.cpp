//===- BufferingStrategy.cpp - Buffer placement strategy ---------- C++ -*-===//
//
// Implements the infrastructure for specifying and manipulating buffering
// strategies.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/BufferingStrategy.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace circt::handshake;
using namespace dynamatic;

/// Returns the index corresponding to the operation result in the operation's
/// results list.
static size_t getResIdx(Operation *op, OpResult res) {
  for (auto [idx, opRes] : llvm::enumerate(op->getResults()))
    if (res == opRes)
      return idx;
  llvm_unreachable("result does not exist");
}

DenseMap<OpResult, ChannelBufProps> dynamatic::getOpBufProps(Operation *op) {
  DenseMap<OpResult, ChannelBufProps> props;

  // Check if the attribute containing the properties is defined
  auto opPropsAttr = op->getAttrOfType<OpBufPropsAttr>(BUF_PROPS_ATTR);
  if (!opPropsAttr)
    return props;

  // Map each result index to the corresponding OpResult
  auto numResults = op->getNumResults();
  for (auto &[resIdx, channelProps] : opPropsAttr.getAllChannelProps()) {
    assert(resIdx < numResults && "resIdx must be < than number of results");
    props.insert(std::make_pair(op->getResult(resIdx), channelProps));
  }
  return props;
}

void dynamatic::clearBufProps(Operation *op) {
  // Check if the attribute containing the properties is defined
  if (!op->hasAttrOfType<OpBufPropsAttr>(BUF_PROPS_ATTR))
    return;

  op->removeAttr(BUF_PROPS_ATTR);
}

LogicalResult dynamatic::addChannelBufProps(OpResult res,
                                            ChannelBufProps channelProps) {
  Operation *op = res.getDefiningOp();
  size_t resIdx = getResIdx(op, res);
  SmallVector<std::pair<size_t, ChannelBufProps>, 4> allChannelProps;

  // Get the attribute if it exists. If yes, copy its existing channel buffering
  // properties and remove it
  if (auto attr = op->getAttrOfType<OpBufPropsAttr>(BUF_PROPS_ATTR)) {
    llvm::copy(attr.getAllChannelProps(), std::back_inserter(allChannelProps));
    op->removeAttr(BUF_PROPS_ATTR);
  }

  // Make sure the channel doesn't already have buffering properties
  for (auto &[resIdxInList, _] : allChannelProps)
    if (resIdx == resIdxInList)
      return op->emitError()
             << "Trying to add channel buffering properties for result "
             << resIdx << ", but some already exist";

  // Add the new channel properties and set the attribute
  allChannelProps.push_back(std::make_pair(resIdx, channelProps));
  op->setAttr(BUF_PROPS_ATTR,
              OpBufPropsAttr::get(res.getContext(), allChannelProps));
  return success();
}

bool dynamatic::replaceChannelBufProps(OpResult res,
                                       ChannelBufProps channelProps) {
  Operation *op = res.getDefiningOp();
  size_t resIdx = getResIdx(op, res);
  SmallVector<std::pair<size_t, ChannelBufProps>, 4> allChannelProps;
  bool replaced = false;

  // Get the attribute if it exists. If yes, copy its existing channel buffering
  // properties while potentially replacing the one given as argument, then
  // remove the attribute
  if (auto attr = op->getAttrOfType<OpBufPropsAttr>(BUF_PROPS_ATTR)) {
    for (auto &[resIdxInList, channelPropsInList] : attr.getAllChannelProps()) {
      if (resIdxInList == resIdx) {
        allChannelProps.push_back(std::make_pair(resIdx, channelProps));
        replaced = true;
      } else
        allChannelProps.push_back(std::make_pair(resIdx, channelPropsInList));
    }
    op->removeAttr(BUF_PROPS_ATTR);
  } else
    allChannelProps.push_back(std::make_pair(resIdx, channelProps));

  op->setAttr(BUF_PROPS_ATTR,
              OpBufPropsAttr::get(res.getContext(), allChannelProps));
  return replaced;
}
