//===- BufferingProperties.cpp - Buffer placement properties ----*- C++ -*-===//
//
// Implements the infrastructure for specifying and manipulating
// channel-specific buffering properties.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/BufferingProperties.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace circt;
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

/// Returns the index corresponding to the block argument in the block's
/// arguments list.
static size_t getArgIdx(Block *block, BlockArgument arg) {
  for (auto [idx, blockArg] : llvm::enumerate(block->getArguments()))
    if (arg == blockArg)
      return idx;
  llvm_unreachable("block argument does not exist");
}

/// Returns buffering properties associated to the index in the provided
/// operation, if any.
static std::optional<ChannelBufProps> getProps(Operation *op, size_t idx) {
  // Check if the attribute containing the properties is defined
  auto opPropsAttr = op->getAttrOfType<OpBufPropsAttr>(buffer::BUF_PROPS_ATTR);
  if (!opPropsAttr)
    return {};

  // Check if buffering properties exist for the value pointed to by the index
  for (auto &[propIdx, channelProps] : opPropsAttr.getAllChannelProps())
    if (propIdx == idx)
      return channelProps;
  return {};
}

/// Adds buffering properties for the value defined by the operation at the
/// provided index. Fails is some buffering properties already exist for this
/// value.
static LogicalResult addProps(Operation *op, size_t idx,
                              ChannelBufProps &props) {
  SmallVector<std::pair<size_t, ChannelBufProps>, 4> allChannelProps;

  // Get the attribute if it exists. If yes, copy its existing channel buffering
  // properties and remove it
  if (auto attr = op->getAttrOfType<OpBufPropsAttr>(buffer::BUF_PROPS_ATTR)) {
    llvm::copy(attr.getAllChannelProps(), std::back_inserter(allChannelProps));
    op->removeAttr(buffer::BUF_PROPS_ATTR);
  }

  // Make sure the channel doesn't already have buffering properties
  for (auto &[resIdxInList, _] : allChannelProps)
    if (idx == resIdxInList)
      return op->emitError()
             << "Trying to add channel buffering properties for result " << idx
             << ", but some already exist";

  // Add the new channel properties and set the attribute
  allChannelProps.push_back(std::make_pair(idx, props));
  op->setAttr(buffer::BUF_PROPS_ATTR,
              OpBufPropsAttr::get(op->getContext(), allChannelProps));
  return success();
}

/// Adds or replaces buffering properties for the value defined by the operation
/// at the provided index.
static void replaceProps(Operation *op, size_t idx, ChannelBufProps &props,
                         bool *replaced) {
  SmallVector<std::pair<size_t, ChannelBufProps>, 4> allChannelProps;

  // Get the attribute if it exists. If yes, copy its existing channel buffering
  // properties while potentially replacing the one given as argument, then
  // remove the attribute
  if (auto attr = op->getAttrOfType<OpBufPropsAttr>(buffer::BUF_PROPS_ATTR)) {
    for (auto &[resIdxInList, channelPropsInList] : attr.getAllChannelProps()) {
      if (resIdxInList == idx) {
        allChannelProps.push_back(std::make_pair(idx, props));
        if (replaced)
          *replaced = true;
      } else
        allChannelProps.push_back(std::make_pair(idx, channelPropsInList));
    }
    op->removeAttr(buffer::BUF_PROPS_ATTR);
  } else
    allChannelProps.push_back(std::make_pair(idx, props));

  // Recreate the entire attribute
  op->setAttr(buffer::BUF_PROPS_ATTR,
              OpBufPropsAttr::get(op->getContext(), allChannelProps));
}

DenseMap<Value, ChannelBufProps>
dynamatic::buffer::getAllBufProps(Operation *op) {
  DenseMap<Value, ChannelBufProps> props;
  // Check if the attribute containing the properties is defined
  auto opPropsAttr = op->getAttrOfType<OpBufPropsAttr>(BUF_PROPS_ATTR);
  if (!opPropsAttr)
    return props;

  if (handshake::FuncOp funcOp = dyn_cast<handshake::FuncOp>(op)) {
    // Map each function argument to the corresponding value
    unsigned numArgs = funcOp.getNumArguments();
    for (auto &[argIDx, channelProps] : opPropsAttr.getAllChannelProps()) {
      assert(argIDx < numArgs && "argIdx must be < than number of results");
      props.insert(std::make_pair(funcOp.getArgument(argIDx), channelProps));
    }
    return props;
  }

  // Map each result index to the corresponding value
  unsigned numResults = op->getNumResults();
  for (auto &[resIdx, channelProps] : opPropsAttr.getAllChannelProps()) {
    assert(resIdx < numResults && "resIdx must be < than number of results");
    props.insert(std::make_pair(op->getResult(resIdx), channelProps));
  }
  return props;
}

std::optional<ChannelBufProps> dynamatic::buffer::getBufProps(OpResult res) {
  Operation *op = res.getDefiningOp();
  return getProps(op, getResIdx(op, res));
}

std::optional<ChannelBufProps>
dynamatic::buffer::getBufProps(BlockArgument arg) {
  Block *block = arg.getParentBlock();
  handshake::FuncOp funcOp = dyn_cast<handshake::FuncOp>(block->getParentOp());
  if (!funcOp)
    return {};
  return getProps(funcOp, getArgIdx(block, arg));
}

void dynamatic::buffer::clearBufProps(Operation *op) {
  // Check if the attribute containing the properties is defined
  if (!op->hasAttrOfType<OpBufPropsAttr>(BUF_PROPS_ATTR))
    return;

  op->removeAttr(BUF_PROPS_ATTR);
}

LogicalResult dynamatic::buffer::addBufProps(OpResult res,
                                             ChannelBufProps &props) {
  Operation *op = res.getDefiningOp();
  return addProps(op, getResIdx(op, res), props);
}

LogicalResult dynamatic::buffer::addBufProps(BlockArgument arg,
                                             ChannelBufProps &props) {
  Block *block = arg.getParentBlock();
  handshake::FuncOp funcOp = dyn_cast<handshake::FuncOp>(block->getParentOp());
  if (!funcOp)
    return funcOp->emitError()
           << "expected block parent to be Handshake function";
  return addProps(funcOp, getArgIdx(block, arg), props);
}

LogicalResult dynamatic::buffer::replaceBufProps(OpResult res,
                                                 ChannelBufProps &props,
                                                 bool *replaced) {
  Operation *op = res.getDefiningOp();
  replaceProps(op, getResIdx(op, res), props, replaced);
  return success();
}

LogicalResult dynamatic::buffer::replaceBufProps(BlockArgument arg,
                                                 ChannelBufProps &props,
                                                 bool *replaced) {
  Block *block = arg.getParentBlock();
  handshake::FuncOp funcOp = dyn_cast<handshake::FuncOp>(block->getParentOp());
  if (!funcOp)
    return funcOp->emitError()
           << "expected block parent to be Handshake function";
  replaceProps(funcOp, getArgIdx(block, arg), props, replaced);
  return success();
}
