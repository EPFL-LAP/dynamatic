//===- BufferingProperties.cpp - Buffer placement properties ----*- C++ -*-===//
//
// Implements the infrastructure for specifying and manipulating
// channel-specific buffering properties.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/BufferingProperties.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/ErrorHandling.h"
#include <string>

using namespace mlir;
using namespace circt;
using namespace circt::handshake;
using namespace dynamatic;
using namespace dynamatic::buffer;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Converts the attribute's name to an unsigned number. If building in debug
/// mode, asserts if the attribute name doesn't represent a valid index.
static inline size_t toIdx(const NamedAttribute &attr) {
  std::string str = attr.getName().str();
  assert(std::all_of(str.begin(), str.end(),
                     [](char c) { return std::isdigit(c); }) &&
         "invalid idx");
  return stoi(str);
}

/// Converts the attribute's value to channel buffering properties. Asserts if
/// the attribte value doesn't represent channel buffering properties.
static inline ChannelBufProps toBufProps(const NamedAttribute &attr) {
  return attr.getValue().cast<ChannelBufPropsAttr>().getProps();
}

/// If the channel is an operation's result or a function argument, saves its
/// index and producing operation. Fails otherwise, in which case the
/// index is undefined and op points to the channel's parent operation.
static LogicalResult getChannelOpAndIdx(Value channel, Operation *&op,
                                        size_t &idx) {
  if (OpResult res = dyn_cast<OpResult>(channel)) {
    op = channel.getDefiningOp();
    idx = res.getResultNumber();
    return success();
  }

  // Channel must be a block argument. In this case we only support buffering
  // properties the channel maps to a Handshake function argument
  BlockArgument arg = cast<BlockArgument>(channel);
  op = dyn_cast<handshake::FuncOp>(arg.getParentBlock()->getParentOp());
  if (!op)
    return failure();
  idx = arg.getArgNumber();
  return success();
}

/// Returns buffering properties associated to a channel, if any are defined.
/// Additionally, if validChannel is not nullptr, sets it to false if the
/// channel cannot have buffering properties attached to it.
static std::optional<ChannelBufProps> getProps(Value channel,
                                               bool *validChannel = nullptr) {
  Operation *op = nullptr;
  size_t idx = 0;
  if (failed(getChannelOpAndIdx(channel, op, idx))) {
    if (validChannel)
      *validChannel = false;
    return {};
  }

  // Check if the attribute containing the properties is defined
  auto opPropsAttr = op->getAttrOfType<OpBufPropsAttr>(BUF_PROPS_ATTR);
  if (!opPropsAttr)
    return {};

  // Look for buffering properties attached to the channel
  for (const NamedAttribute &attr : opPropsAttr.getChannelProperties())
    if (toIdx(attr) == idx)
      return toBufProps(attr);
  return {};
}

/// Adds or replaces a set of channel buffering properties in the map maintained
/// by the channel's defining operation. Fails if the channel is not
/// bufferizable (i.e, not an operation's result or a function argument), or if
/// stopOnOverwrite is true and buffering properties already exist for the
/// channel. Otherwise, succeeds and sets the last argument to true if it is not
/// nullptr and buffering properties already exist for the channel
static LogicalResult setProps(Value channel, ChannelBufProps &props,
                              bool stopOnOverwrite, bool *replaced = nullptr) {
  Operation *op = nullptr;
  size_t idx = 0;
  if (failed(getChannelOpAndIdx(channel, op, idx)))
    return op->emitError() << "expected block argument's parent operation "
                              "to be Handshake function";

  MLIRContext *ctx = channel.getContext();
  SmallVector<NamedAttribute> channelProperties;
  channelProperties.push_back(
      NamedAttribute(StringAttr::get(ctx, std::to_string(idx)),
                     ChannelBufPropsAttr::get(ctx, props)));

  // Get the attribute if it exists. If yes, copy its existing channel buffering
  // properties while potentially replacing the one given as argument, then
  // remove the attribute
  if (auto attr = op->getAttrOfType<OpBufPropsAttr>(buffer::BUF_PROPS_ATTR)) {
    for (const NamedAttribute &attr : attr.getChannelProperties()) {
      if (toIdx(attr) == idx) {
        if (replaced)
          *replaced = true;
        if (stopOnOverwrite)
          return failure();
      } else
        channelProperties.push_back(attr);
    }
    op->removeAttr(buffer::BUF_PROPS_ATTR);
  }

  // Recreate the entire attribute
  op->setAttr(buffer::BUF_PROPS_ATTR,
              OpBufPropsAttr::get(ctx, channelProperties));
  return success();
}

//===----------------------------------------------------------------------===//
// LazyBufProps implementation
//===----------------------------------------------------------------------===//

LazyChannelBufProps::LazyChannelBufProps(Value val) : val(val){};

bool LazyChannelBufProps::updateIR() {
  bool updated = updateIRIfNecessary();
  if (updated)
    unchangedProps = props;
  return updated;
}

ChannelBufProps &LazyChannelBufProps::operator*() {
  if (!props.has_value())
    readAttribute();
  return props.value();
  ;
}

ChannelBufProps *LazyChannelBufProps::operator->() {
  if (!props.has_value())
    readAttribute();
  return &props.value();
}

LazyChannelBufProps::~LazyChannelBufProps() { updateIRIfNecessary(); }

void LazyChannelBufProps::readAttribute() {
  bool validChannel = true;
  std::optional<ChannelBufProps> optProps = getProps(val, &validChannel);
  assert(validChannel && "channel cannot have buffering properties attached");
  if (optProps.has_value())
    props = optProps.value();
  else
    props = ChannelBufProps();
  unchangedProps = props;
}

bool LazyChannelBufProps::updateIRIfNecessary() {
  if (!props.has_value() || props.value() == unchangedProps.value())
    return false;
  assert(succeeded(buffer::replaceBufProps(val, props.value())) &&
         "failed to replace buffering properties");
  return true;
}

//===----------------------------------------------------------------------===//
// Public functions to manipulate channel buffering properties
//===----------------------------------------------------------------------===//

DenseMap<Value, ChannelBufProps>
dynamatic::buffer::getAllBufProps(Operation *op) {
  DenseMap<Value, ChannelBufProps> props;
  // Check if the attribute containing the properties is defined
  auto opPropsAttr = op->getAttrOfType<OpBufPropsAttr>(BUF_PROPS_ATTR);
  if (!opPropsAttr)
    return props;

  if (handshake::FuncOp funcOp = dyn_cast<handshake::FuncOp>(op)) {
    // Map each function argument to the corresponding value
    for (const NamedAttribute &attr : opPropsAttr.getChannelProperties())
      props.insert(
          std::make_pair(funcOp.getArgument(toIdx(attr)), toBufProps(attr)));
    return props;
  }

  // Map each result index to the corresponding value
  for (const NamedAttribute &attr : opPropsAttr.getChannelProperties())
    props.insert(std::make_pair(op->getResult(toIdx(attr)), toBufProps(attr)));

  return props;
}

std::optional<ChannelBufProps> dynamatic::buffer::getBufProps(Value channel) {
  return getProps(channel, nullptr);
}

LogicalResult dynamatic::buffer::addBufProps(Value channel,
                                             ChannelBufProps &props) {
  bool replaced = false;
  return failure(failed(setProps(channel, props, true, &replaced)) || replaced);
}

LogicalResult dynamatic::buffer::replaceBufProps(Value channel,
                                                 ChannelBufProps &props,
                                                 bool *replaced) {
  return setProps(channel, props, false, replaced);
}

void dynamatic::buffer::clearBufProps(Operation *op) {
  // Check if the attribute containing the properties is defined
  if (!op->hasAttrOfType<OpBufPropsAttr>(BUF_PROPS_ATTR))
    return;

  op->removeAttr(BUF_PROPS_ATTR);
}
