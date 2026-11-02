//===- CfToHandshake.cpp - cf to handshake, per-consumer ------------------===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowers cf to handshake by routing. A routing delivers a value from the
// blocks that produces it to a block that uses it. A memory interface is a
// routing too: its single memory token goes from the memory's start through
// every block that accesses the interface to the memory's end, which
// sequentializes the interface's control signals. Every routing gets its own
// control network, built on the function's cfg but optimized to that routing
// alone, so that nothing waits on a control-flow decision that does not
// concern it.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Conversion/CfToHandshake/AnnotatedCfg.h"
#include "dynamatic/Conversion/CfToHandshake/BaseCfg.h"
#include "dynamatic/Conversion/CfToHandshake/ControlNetworkRouting.h"
#include "dynamatic/Conversion/CfToHandshake/TempOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/BufferPlacement/Utils/BufferingSupport.h"
#include "dynamatic/Transforms/FuncMaximizeSSA.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

// [START Boilerplate code for the MLIR pass]
#include "dynamatic/Conversion/Passes.h" // IWYU pragma: keep
namespace dynamatic {
#define GEN_PASS_DEF_CFTOHANDSHAKE
#include "dynamatic/Conversion/Passes.h.inc"
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental::cf_to_handshake;
using dynamatic::buffer::Channel;

/// We refer to a handshake load or store
/// as a memory access
/// (lucas used to call them memory ports, but this is very ambiguous)
using MemoryAccess = handshake::MemPortOpInterface;

namespace {

/// The handshake type a cf-level type lowers to.
///
/// MLIR's index type has no fixed width,
/// but a channel only carries integers and floats.
/// The rest of the lowering already treats index as 32 bits wide
/// (index constants and index casts),
/// so index lowers to a 32-bit integer here too, to match.
static Type getHandshakeType(Type type) {
  MLIRContext *ctx = type.getContext();
  Type i32Type = IntegerType::get(ctx, 32);

  // an index value becomes a channel of a 32-bit integer
  if (isa<IndexType>(type)) {
    return handshake::ChannelType::get(i32Type);
  }

  // integer and float values become channels of themselves
  if (isa<IntegerType, FloatType>(type)) {
    return handshake::ChannelType::get(type);
  }

  // a memory region (an array argument) is not routed:
  // it stays a memref, for the memory interfaces
  if (auto memrefType = dyn_cast<MemRefType>(type)) {
    // an array of anything but index is unchanged
    if (!isa<IndexType>(memrefType.getElementType())) {
      return memrefType;
    }

    // an array of index has its loads produce index values,
    // which become channels, so they must be 32-bit integers:
    // the array's elements become 32-bit integers to match
    return MemRefType::get(memrefType.getShape(), i32Type);
  }

  // handshake types are already lowered
  if (isa<handshake::ChannelType, handshake::ControlType>(type)) {
    return type;
  }

  llvm_unreachable("type has no handshake lowering");
}

// Struct to store information about a lowered function
// The function operation itself
//
// The dataless "start" function arg
// that is added on top of the program-level function args
//
// And a mapping from memrefs to memStarts
// which are dataless function args added, each of which
// starts execution on a memory region
// (allowing loads and stores from a specific array)
struct HandshakeFunction {
  handshake::FuncOp funcOp;
  Value start;
  MapVector<MemrefValue, ControlValue> memStarts;
};

/// Creates the handshake function, and moves the body into it.
///
/// It takes the original arguments in their handshake types,
/// a start control per memory region, and the start control.
/// It returns the original results as channels,
/// an end control per memory region, and the end control.
///
/// The terminators stay in place,
/// converted instead at the end of the pass.
static HandshakeFunction createHandshakeFunction(func::FuncOp funcOp,
                                                 NameAnalysis &namer,
                                                 OpBuilder &builder) {
  // Handle the MLIR context
  MLIRContext *ctx = builder.getContext();

  // Handle MLIR type stuff, getting the control type
  // used to make the extra dataless function args
  // and returned values
  // (start and end, including memStarts and memEnds)
  Type controlType = handshake::ControlType::get(ctx);

  // We use the function arg names from the C input
  // to name the handshaked versions
  //
  // Results in C don't have names,
  // so we just name the original results out0 to outN
  //
  // arg and res names are attributes as they are the final names
  // that go directly as attributes on the handshake function op
  SmallVector<Attribute> argNames;
  SmallVector<Attribute> resNames;

  // We also use the names of the memory regions
  // (the memref args, which become array inputs, and the RAM ops)
  // to produce the names of the extra mem start args and mem end results.
  // we store the mem names as strings as the memref name doesn't correspond
  // directly to the attributes we make from them
  // then the attribute made goes on the handshake function op
  //
  // Also store the memrefs directly
  // as we need to add a control for each of them to the entry block also
  SmallVector<std::string> memNames;
  SmallVector<MemrefValue> memrefs;

  // for each function arg
  for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
    // gets its name from the attr
    auto name = funcOp.getArgAttrOfType<StringAttr>(idx, "handshake.arg_name");
    assert(name && "function argument has no name");
    // store the arg name
    argNames.push_back(name);
    // if its an array
    if (auto memref = dyn_cast<MemrefValue>(arg)) {
      // store in mem names also
      memNames.push_back(name.str());
      memrefs.push_back(memref);
    }
  }

  // for each RAM op
  for (auto ramOp : funcOp.getBody().getOps<handshake::RAMOp>()) {
    // store in mem names also
    memNames.push_back(namer.getName(ramOp).str());
    memrefs.push_back(cast<MemrefValue>(ramOp.getResult()));
  }

  // every memory region must be statically sized and one-dimensional
  for (MemrefValue memref : memrefs) {
    assert(memref.getType().hasStaticShape() &&
           memref.getType().getRank() == 1 &&
           "memory region must be statically sized and one-dimensional");
  }

  // after them come a start control per memory region, then the start
  // for each memory region
  for (const std::string &memName : memNames) {
    // name its start control after it
    argNames.push_back(StringAttr::get(ctx, memName + "_start"));
  }
  // and name the start control last
  argNames.push_back(StringAttr::get(ctx, "start"));

  // for each original result
  for (unsigned idx = 0; idx < funcOp.getNumResults(); ++idx) {
    // give it a default name by its position
    resNames.push_back(StringAttr::get(ctx, "out" + std::to_string(idx)));
  }
  // for each memory region
  for (const std::string &memName : memNames) {
    // name its end control after it
    resNames.push_back(StringAttr::get(ctx, memName + "_end"));
  }
  // and name the end control last
  resNames.push_back(StringAttr::get(ctx, "end"));

  // the original arguments and results take their handshake types,
  // and every control is dataless
  SmallVector<Type> argTypes, resTypes;
  // for each original argument, its channel type
  for (Type type : funcOp.getArgumentTypes()) {
    argTypes.push_back(getHandshakeType(type));
  }
  // for each original result, its channel type
  for (Type type : funcOp.getResultTypes()) {
    resTypes.push_back(getHandshakeType(type));
  }
  // a start control per memory region
  argTypes.append(memNames.size(), controlType);
  // and the start
  argTypes.push_back(controlType);

  // an end control per memory region
  resTypes.append(memNames.size(), controlType);
  // and the end
  resTypes.push_back(controlType);

  // Make Named Attributes that go from
  // argNames -> arg name list
  // and
  // resNames -> res name list
  SmallVector<NamedAttribute> attrs;
  attrs.emplace_back(StringAttr::get(ctx, "argNames"),
                     ArrayAttr::get(ctx, argNames));
  attrs.emplace_back(StringAttr::get(ctx, "resNames"),
                     ArrayAttr::get(ctx, resNames));

  // create the handshake function next to the original
  builder.setInsertionPoint(funcOp);
  auto newFuncOp = handshake::FuncOp::create(
      builder, funcOp.getLoc(), funcOp.getName(),
      builder.getFunctionType(argTypes, resTypes), attrs);

  // Make the struct that we will return
  // that stores all the info about the handshake func op
  HandshakeFunction handshakeFunctionInfo;
  handshakeFunctionInfo.funcOp = newFuncOp;

  // and move the body of the func.func over to the handshake.func
  newFuncOp.getBody().takeBody(funcOp.getBody());

  // convert every block argument to its handshake type
  for (Block &block : newFuncOp.getBody()) {
    for (BlockArgument arg : block.getArguments()) {
      arg.setType(getHandshakeType(arg.getType()));
    }
  }

  // in MLIR, a function's arguments are its entry block's arguments,
  // which must match its signature.
  // The extra arg names we added in the arg names attribute
  // must also be added as block args of the correct type to the entry block
  Block &entry = newFuncOp.getBody().front();

  // add a start control arg to the entry block per memory region
  for (MemrefValue memref : memrefs) {
    handshakeFunctionInfo.memStarts[memref] =
        cast<ControlValue>(entry.addArgument(controlType, funcOp.getLoc()));
  }

  // and the start control arg to the entry block
  handshakeFunctionInfo.start = entry.addArgument(controlType, funcOp.getLoc());

  return handshakeFunctionInfo;
}

/// Whether a constant can be triggered by an always-valid source,
/// rather than by start.
///
/// A source triggers the constant whenever its consumers are ready.
/// Some consumers need exactly one token per execution of the constant's
/// block, and then the constant must be triggered by start, routed to its
/// block.
static bool isSourcable(Operation *op) {
  // only a constant can be sourced:
  // uninitialized variables are triggered by start
  if (!isa<arith::ConstantOp>(op)) {
    return false;
  }

  // inherited list of the ops that need this
  for (Operation *user : op->getUsers()) {
    if (isa<cf::BranchOp, cf::CondBranchOp, memref::LoadOp, memref::StoreOp>(
            user)) {
      return false;
    }
  }

  return true;
}

/// The value a constant produces.
/// An index constant lowers to a 32-bit integer,
/// as index does everywhere in this pass (see getHandshakeType).
static TypedAttr getConstantValue(arith::ConstantOp cstOp, OpBuilder &builder) {
  TypedAttr value = cstOp.getValue();
  if (!isa<IndexType>(value.getType())) {
    return value;
  }
  return IntegerAttr::get(builder.getIntegerType(32),
                          cast<IntegerAttr>(value).getValue().trunc(32));
}

/// The value an undefined value produces:
/// zero, in its type, with index lowered to a 32-bit integer.
/// Returns null for a type with no zero.
static TypedAttr getUndefinedValue(LLVM::UndefOp undefOp, OpBuilder &builder) {
  Type type = undefOp.getRes().getType();
  if (isa<IndexType>(type)) {
    return builder.getIntegerAttr(builder.getIntegerType(32), 0);
  }
  if (isa<IntegerType>(type)) {
    return builder.getIntegerAttr(type, 0);
  }
  if (auto floatType = dyn_cast<FloatType>(type)) {
    return builder.getFloatAttr(floatType, 0.0);
  }
  return nullptr;
}

/// Converts the constants and the undefined values to handshake constants.
///
/// A sourcable constant is triggered by a source.
/// Any other constant, and every undefined value, is triggered by start,
/// which makes it a consumer of start that the routing delivers to its block.
///
/// Returns failure after emitting an error
/// for an undefined value of a type with no zero.
static LogicalResult convertConstants(Region &body, Value start,
                                      OpBuilder &builder) {
  // walk every operation, in every block
  for (Block &block : body) {
    for (Operation &op : llvm::make_early_inc_range(block)) {
      // only constants and undefined values are converted here
      if (!isa<arith::ConstantOp, LLVM::UndefOp>(op)) {
        continue;
      }
      Location loc = op.getLoc();
      builder.setInsertionPoint(&op);

      // the value the handshake constant produces
      TypedAttr value;
      if (auto cstOp = dyn_cast<arith::ConstantOp>(op)) {
        value = getConstantValue(cstOp, builder);
      } else {
        value = getUndefinedValue(cast<LLVM::UndefOp>(op), builder);
      }
      if (!value) {
        return op.emitError() << "operation has unsupported result type";
      }

      // what triggers it: a source, or start
      Value trigger = start;
      if (isSourcable(&op)) {
        trigger = handshake::SourceOp::create(builder, loc).getResult();
      }

      // replace the original with the handshake constant,
      // which takes over its uses
      auto newOp = handshake::ConstantOp::create(builder, loc, value, trigger);
      op.replaceAllUsesWith(newOp);
      op.erase();
    }
  }
  return success();
}

/// The type a channel carries, or the type itself if it is not a channel.
static Type getDataType(Type type) {
  if (auto channelType = dyn_cast<handshake::ChannelType>(type)) {
    return channelType.getDataType();
  }
  return type;
}

/// Replaces an arith or math operation with its handshake equivalent, built at
/// the builder's position with the given result types and the operation's
/// operands and attributes, and erases it. An index cast that does not change
/// the width is replaced by its operand. Fails for an operation with no
/// handshake equivalent.
static LogicalResult replaceArithOp(Operation &op, TypeRange resultTypes,
                                    OpBuilder &builder) {
  Location loc = op.getLoc();
  ValueRange operands = op.getOperands();
  ArrayRef<NamedAttribute> attrs = op.getAttrs();

  // build the handshake op, getting the value replacing the op's result
  auto buildReplacement = [&]() -> FailureOr<Value> {
    auto indexCastOp = dyn_cast<arith::IndexCastOp>(op);
    auto indexCastUIOp = dyn_cast<arith::IndexCastUIOp>(op);
    if (indexCastOp || indexCastUIOp) {
      Value in = indexCastOp ? indexCastOp.getIn() : indexCastUIOp.getIn();
      Value out = indexCastOp ? indexCastOp.getOut() : indexCastUIOp.getOut();

      // if its an index cast op,
      // its from an integer going to index an array
      assert(isa<IndexType>(out.getType()) &&
             "index cast from index to an integer");
      // get the integer width
      unsigned srcWidth = getDataType(in.getType()).getIntOrFloatBitWidth();

      // index types are treated as i32 currently
      if (srcWidth == 32) {
        // if the incoming integer is i32
        // just use it directly
        return in;
      }

      if (srcWidth > 32) {
        // if its wider, we need to truncate
        return handshake::TruncIOp::create(builder, loc, resultTypes, operands,
                                           attrs)
            .getOut();
      }
      if (indexCastOp) {
        // if its narrower, either sign extend
        return handshake::ExtSIOp::create(builder, loc, resultTypes, operands,
                                          attrs)
            .getOut();
      }
      // or unsign extend
      return handshake::ExtUIOp::create(builder, loc, resultTypes, operands,
                                        attrs)
          .getOut();
    }

    // every other operation has a handshake operation of the same kind
    return llvm::TypeSwitch<Operation *, FailureOr<Value>>(&op)
        .Case<arith::AddFOp>([&](auto) {
          return handshake::AddFOp::create(builder, loc, resultTypes, operands,
                                           attrs)
              .getResult();
        })
        .Case<arith::AddIOp>([&](auto) {
          return handshake::AddIOp::create(builder, loc, resultTypes, operands,
                                           attrs)
              .getResult();
        })
        .Case<arith::AndIOp>([&](auto) {
          return handshake::AndIOp::create(builder, loc, resultTypes, operands,
                                           attrs)
              .getResult();
        })
        .Case<arith::CmpFOp>([&](arith::CmpFOp cmpFOp) {
          // handshake's predicates have the same values as arith's
          auto predicate =
              static_cast<handshake::CmpFPredicate>(cmpFOp.getPredicate());
          return handshake::CmpFOp::create(
                     builder, loc,
                     handshake::ChannelType::get(builder.getI1Type()),
                     predicate, cmpFOp.getLhs(), cmpFOp.getRhs())
              .getResult();
        })
        .Case<arith::CmpIOp>([&](arith::CmpIOp cmpIOp) {
          // handshake's predicates have the same values as arith's
          auto predicate =
              static_cast<handshake::CmpIPredicate>(cmpIOp.getPredicate());
          return handshake::CmpIOp::create(
                     builder, loc,
                     handshake::ChannelType::get(builder.getI1Type()),
                     predicate, cmpIOp.getLhs(), cmpIOp.getRhs())
              .getResult();
        })
        .Case<arith::DivFOp>([&](auto) {
          return handshake::DivFOp::create(builder, loc, resultTypes, operands,
                                           attrs)
              .getResult();
        })
        .Case<arith::DivSIOp>([&](auto) {
          return handshake::DivSIOp::create(builder, loc, resultTypes, operands,
                                            attrs)
              .getResult();
        })
        .Case<arith::DivUIOp>([&](auto) {
          return handshake::DivUIOp::create(builder, loc, resultTypes, operands,
                                            attrs)
              .getResult();
        })
        .Case<arith::RemSIOp>([&](auto) {
          return handshake::RemSIOp::create(builder, loc, resultTypes, operands,
                                            attrs)
              .getResult();
        })
        .Case<arith::RemUIOp>([&](auto) {
          return handshake::RemUIOp::create(builder, loc, resultTypes, operands,
                                            attrs)
              .getResult();
        })
        .Case<arith::ExtSIOp>([&](auto) {
          return handshake::ExtSIOp::create(builder, loc, resultTypes, operands,
                                            attrs)
              .getOut();
        })
        .Case<arith::ExtUIOp>([&](auto) {
          return handshake::ExtUIOp::create(builder, loc, resultTypes, operands,
                                            attrs)
              .getOut();
        })
        .Case<arith::MaximumFOp>([&](auto) {
          return handshake::MaximumFOp::create(builder, loc, resultTypes,
                                               operands, attrs)
              .getResult();
        })
        .Case<arith::MinimumFOp>([&](auto) {
          return handshake::MinimumFOp::create(builder, loc, resultTypes,
                                               operands, attrs)
              .getResult();
        })
        .Case<arith::MaxSIOp>([&](auto) {
          return handshake::MaxSIOp::create(builder, loc, resultTypes, operands,
                                            attrs)
              .getResult();
        })
        .Case<arith::MaxUIOp>([&](auto) {
          return handshake::MaxUIOp::create(builder, loc, resultTypes, operands,
                                            attrs)
              .getResult();
        })
        .Case<arith::MinSIOp>([&](auto) {
          return handshake::MinSIOp::create(builder, loc, resultTypes, operands,
                                            attrs)
              .getResult();
        })
        .Case<arith::MinUIOp>([&](auto) {
          return handshake::MinUIOp::create(builder, loc, resultTypes, operands,
                                            attrs)
              .getResult();
        })
        .Case<arith::MulFOp>([&](auto) {
          return handshake::MulFOp::create(builder, loc, resultTypes, operands,
                                           attrs)
              .getResult();
        })
        .Case<arith::MulIOp>([&](auto) {
          return handshake::MulIOp::create(builder, loc, resultTypes, operands,
                                           attrs)
              .getResult();
        })
        .Case<arith::NegFOp>([&](auto) {
          return handshake::NegFOp::create(builder, loc, resultTypes, operands,
                                           attrs)
              .getOuts();
        })
        .Case<arith::OrIOp>([&](auto) {
          return handshake::OrIOp::create(builder, loc, resultTypes, operands,
                                          attrs)
              .getResult();
        })
        .Case<arith::SelectOp>([&](arith::SelectOp selectOp) {
          return handshake::SelectOp::create(
                     builder, loc, getHandshakeType(selectOp.getType()),
                     selectOp.getCondition(), selectOp.getTrueValue(),
                     selectOp.getFalseValue(),
                     /*antitokenDepth=*/IntegerAttr())
              .getResult();
        })
        .Case<arith::ShLIOp>([&](auto) {
          return handshake::ShLIOp::create(builder, loc, resultTypes, operands,
                                           attrs)
              .getResult();
        })
        .Case<arith::ShRSIOp>([&](auto) {
          return handshake::ShRSIOp::create(builder, loc, resultTypes, operands,
                                            attrs)
              .getResult();
        })
        .Case<arith::ShRUIOp>([&](auto) {
          return handshake::ShRUIOp::create(builder, loc, resultTypes, operands,
                                            attrs)
              .getResult();
        })
        .Case<arith::SubFOp>([&](auto) {
          return handshake::SubFOp::create(builder, loc, resultTypes, operands,
                                           attrs)
              .getResult();
        })
        .Case<arith::SubIOp>([&](auto) {
          return handshake::SubIOp::create(builder, loc, resultTypes, operands,
                                           attrs)
              .getResult();
        })
        .Case<arith::TruncIOp>([&](auto) {
          return handshake::TruncIOp::create(builder, loc, resultTypes,
                                             operands, attrs)
              .getOut();
        })
        .Case<arith::TruncFOp>([&](auto) {
          return handshake::TruncFOp::create(builder, loc, resultTypes,
                                             operands, attrs)
              .getOut();
        })
        .Case<arith::XOrIOp>([&](auto) {
          return handshake::XOrIOp::create(builder, loc, resultTypes, operands,
                                           attrs)
              .getResult();
        })
        .Case<arith::SIToFPOp>([&](auto) {
          return handshake::SIToFPOp::create(builder, loc, resultTypes,
                                             operands, attrs)
              .getOut();
        })
        .Case<arith::UIToFPOp>([&](auto) {
          return handshake::UIToFPOp::create(builder, loc, resultTypes,
                                             operands, attrs)
              .getOut();
        })
        .Case<arith::FPToSIOp>([&](auto) {
          return handshake::FPToSIOp::create(builder, loc, resultTypes,
                                             operands, attrs)
              .getOut();
        })
        .Case<arith::ExtFOp>([&](auto) {
          return handshake::ExtFOp::create(builder, loc, resultTypes, operands,
                                           attrs)
              .getOut();
        })
        .Case<math::AbsFOp>([&](auto) {
          return handshake::AbsFOp::create(builder, loc, resultTypes, operands,
                                           attrs)
              .getOut();
        })
        .Case<math::AbsIOp>([&](auto) {
          return handshake::AbsIOp::create(builder, loc, resultTypes, operands,
                                           attrs)
              .getOut();
        })
        .Default([&](Operation *) -> FailureOr<Value> {
          op.emitError() << "arith operation has no handshake equivalent";
          return failure();
        });
  };
  FailureOr<Value> replacement = buildReplacement();

  // if we dont get a replacement handshake channel,
  // conversion to handshake has failed
  if (failed(replacement)) {
    return failure();
  }

  // rewire the IR that all consumers of the original
  // read from the replacement handshake channel
  op.replaceAllUsesWith(ValueRange{*replacement});
  // and delete the arith/math op
  op.erase();
  return success();
}

/// Converts every arith and math operation in the body to its handshake
/// equivalent (see replaceArithOp), with handshake result types.
static LogicalResult convertArithAndMath(Region &body, OpBuilder &builder) {
  // for each basic block
  for (Block &block : body) {
    // for each op
    // when we convert an op, we invalidate the iterator, so use early inc
    for (Operation &op : llvm::make_early_inc_range(block)) {
      // we are only converting the arith and math ops in this step
      if (!isa<arith::ArithDialect, math::MathDialect>(op.getDialect())) {
        continue;
      }

      // handle MLIR builder stuff
      builder.setInsertionPoint(&op);

      // the result types of the handshake op
      SmallVector<Type> resultTypes;
      for (Type type : op.getResultTypes()) {
        resultTypes.push_back(getHandshakeType(type));
      }

      // replace the op with its handshake equivalent
      if (failed(replaceArithOp(op, resultTypes, builder))) {
        return failure();
      }
    }
  }
  return success();
}

/// Converts the loads and stores to handshake loads and stores, recorded per
/// memref in program order.
/// A load's data comes from its interface (memory controller or LSQ),
/// which is built after routing,
/// so each load gets a temporary producer for it until then.
static void convertLoadsAndStores(
    Region &body,
    DenseMap<MemrefValue, SmallVector<MemoryAccess>> &memrefToAccesses,
    SmallVector<TempProd> &loadDataTemps, NameAnalysis &namer,
    OpBuilder &builder) {
  // for each basic block
  for (Block &block : body) {
    // for each load and store
    // (walked in program order, as memrefToAccesses needs)
    // converting loads and stores breaks a normal iterator
    for (Operation &op : llvm::make_early_inc_range(block)) {
      // only loads and stores
      if (!isa<memref::LoadOp, memref::StoreOp>(op)) {
        continue;
      }
      // handle insertion point
      builder.setInsertionPoint(&op);

      if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
        // for load ops-

        // our load ops have one index so need to be
        // one-dimensional
        assert(loadOp.getIndices().size() == 1 &&
               "load must be unidimensional");
        // get the address input
        Value address = loadOp.getIndices().front();

        // convert the output type of the load op
        Type dataType = getHandshakeType(loadOp.getResult().getType());

        // temporary producer standing in for the loaded data
        // returning from memory through the MC or LSQ
        TempProd dataTemp(builder, dataType);

        // keep the temp producer purely so we can erase it later
        // (the MC and LSQ builders directly mess with the load ops
        // in a way they shouldn't, so the normal resolution approach
        // with TempProd operations isn't used)
        loadDataTemps.push_back(dataTemp);

        // make the handshake load op
        auto loadHandshake = handshake::LoadOp::create(
            builder, op.getLoc(), address, dataTemp.getValue());

        // copy across the annotations from the memory analysis passes
        copyDialectAttr<handshake::MemDependenceArrayAttr>(&op, loadHandshake);
        copyDialectAttr<handshake::MemInterfaceAttr>(&op, loadHandshake);

        // copy across the name of the load op
        // as the results of the memory analysis passes rely on the op names
        namer.replaceOp(&op, loadHandshake);

        // wire the IR for all consumers of the original load
        // to use the handshake load
        loadOp.getResult().replaceAllUsesWith(loadHandshake.getDataResult());

        // and store that this handshake load op belongs to the memref
        // so we can build the MC/LSQ later
        memrefToAccesses[cast<MemrefValue>(loadOp.getMemRef())].push_back(
            loadHandshake);

        // and delete the memref load op
        loadOp.erase();
      } else {
        // for store ops-
        auto storeOp = cast<memref::StoreOp>(op);

        // our store ops have one index so need to be
        // one-dimensional
        assert(storeOp.getIndices().size() == 1 &&
               "store must be unidimensional");
        // get the address input
        Value address = storeOp.getIndices().front();

        // make the handshake store op
        auto storeHandshake = handshake::StoreOp::create(
            builder, op.getLoc(), address, storeOp.getValueToStore());

        // copy across the annotations from the memory analysis passes
        copyDialectAttr<handshake::MemDependenceArrayAttr>(&op, storeHandshake);
        copyDialectAttr<handshake::MemInterfaceAttr>(&op, storeHandshake);

        // copy across the name of the store op
        // as the results of the memory analysis passes rely on the op names
        namer.replaceOp(&op, storeHandshake);

        // and store that this handshake store op belongs to the memref
        // so we can build the MC/LSQ later
        memrefToAccesses[cast<MemrefValue>(storeOp.getMemRef())].push_back(
            storeHandshake);

        // and delete the memref store op
        storeOp.erase();
      }
    }
  }
}

/// Often,
/// memory controls come out of a lazy fork, whose other output passes the
/// memory token on. That continuing output gets one buffer slot of capacity,
/// so the lazy fork can give the control to the interface without waiting for
/// the next consumer of the token to be ready.
/// Otherwise, it means, due to the CFG,
/// this is one of the possible very last memory controls.
/// Since the memory control token doesn't continue on, there's no lazy fork
static void bufferContinuingTokens(const DenseMap<unsigned, Value> &controls) {
  for (auto [blockNumber, control] : controls) {
    // if ther's a lazy fork
    auto forkOp = control.getDefiningOp<handshake::LazyForkOp>();
    if (!forkOp) {
      continue;
    }
    assert(forkOp->getNumResults() == 2 &&
           "memory control lazy fork must have two outputs");

    // the continuing output gets
    // at least one buffer slot of capacity
    for (Value result : forkOp->getResults()) {
      // the continuing output is the one
      // thats not the memory control
      if (result == control) {
        continue;
      }
      Channel channel(result, /*updateProps=*/true);
      channel.props->minTrans = std::max(channel.props->minTrans, 1U);
    }
  }
}

/// Converts each cf internal array op (alloca and get_global) to a RAM op
static LogicalResult convertInternalArrays(func::FuncOp funcOp,
                                           NameAnalysis &namer,
                                           OpBuilder &builder) {
  for (Block &block : funcOp.getBody()) {
    // converting ops breaks a normal iterator
    for (Operation &op : llvm::make_early_inc_range(block)) {
      // only cf internal array ops get converted to RAM ops
      if (!isa<memref::AllocaOp, memref::GetGlobalOp>(op)) {
        continue;
      }

      // lambda for converting the cf internal array op to a RAM op
      auto makeRamOp = [&]() -> FailureOr<handshake::RAMOp> {
        if (auto allocaOp = dyn_cast<memref::AllocaOp>(op)) {
          // local array

          MemRefType type = allocaOp.getType();
          // a local array starts as all zeros
          auto zeros = DenseElementsAttr::get(
              type, builder.getZeroAttr(type.getElementType()));
          // make the RAM op holding all zeros
          return handshake::RAMOp::create(builder, op.getLoc(), type, zeros);
        } else if (auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(op)) {
          // read-only global array

          // find the global the get_global op reads
          auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
              getGlobalOp, getGlobalOp.getNameAttr());
          if (!global) {
            return getGlobalOp->emitError() << "reference to unknown global";
          }
          // A const global array becomes an internal read-only memory unit.
          // A non-const global array is rejected:
          // it is not guaranteed to be reset to an initial value per
          // invocation, which could imply state across invocations of the
          // kernel, which we do not currently support
          if (!global.getConstant()) {
            return global->emitError()
                   << "a global must be constant to become a RAM";
          }
          // a global array starts as its initial value
          auto initialValue = dyn_cast_if_present<DenseElementsAttr>(
              global.getInitialValueAttr());
          if (!initialValue) {
            return global->emitError()
                   << "the initial value must be denoted in DenseElementsAttr";
          }
          // make the RAM op holding the global's initial value
          return handshake::RAMOp::create(builder, op.getLoc(),
                                          getGlobalOp.getType(), initialValue);
        }
        llvm_unreachable("op is not a cf internal array op");
      };

      // make the RAM op in place of the cf internal array op
      builder.setInsertionPoint(&op);
      FailureOr<handshake::RAMOp> ramOp = makeRamOp();
      if (failed(ramOp)) {
        return failure();
      }

      // name the RAM op
      namer.setName(*ramOp);

      // the loads and stores now access the RAM op
      op.replaceAllUsesWith(*ramOp);
      op.erase();
    }
  }
  return success();
}

/// Lowers one function to handshake.
static LogicalResult lowerFunction(func::FuncOp funcOp, NameAnalysis &namer,
                                   OpBuilder &builder) {
  // point at the function op if anything goes wrong
  Location loc = funcOp.getLoc();

  // this pass supports only functions with a single return
  // we transform functions with multiple returns in an earlier pass
  auto returnOps = funcOp.getOps<func::ReturnOp>();
  assert(std::distance(returnOps.begin(), returnOps.end()) == 1 &&
         "function must have a single return");
  func::ReturnOp returnOp = *returnOps.begin();

  // map each basic block to its branch condition, for routing.
  // this reads the cf branch conditions, which must still be i1: a branch
  // condition can be a function argument, which becomes a channel when
  // the handshake function is created, so this comes first
  BlockToCond blockToCond(funcOp.getBody(), builder);

  // create the handshake function op
  // and get the struct containing the info on it
  HandshakeFunction handshakeFunctionInfo =
      createHandshakeFunction(funcOp, namer, builder);

  // pull the function op and body from the info
  handshake::FuncOp newFuncOp = handshakeFunctionInfo.funcOp;
  Region &body = newFuncOp.getBody();

  // pull the start and memStart inputs from the info
  Value start = handshakeFunctionInfo.start;
  MapVector<MemrefValue, ControlValue> &memStarts =
      handshakeFunctionInfo.memStarts;

  // build the base CFG from the IR
  Cfg base = getBaseCfg(body);

  // We need to collect the routing CFGs for the memory interfaces before we go
  // from memref loads and stores to handshake loads and stores
  // because we find the loads and stores of each memory region
  // by walking the memref
  MapVector<std::pair<MemrefValue, McOrLsq>, MemoryInterfaceCfg> memoryCfgs =
      collectMemoryInterfaceCfgs(base, memStarts);

  // convert constants
  if (failed(convertConstants(body, start, builder))) {
    return failure();
  }
  // convert arith and math
  if (failed(convertArithAndMath(body, builder))) {
    return failure();
  }

  // convert loads and stores
  // storing which handshake loads and stores belong to which memref
  // and storing the temporary data producers added
  // which stand in for loaded data returning from memory
  DenseMap<MemrefValue, SmallVector<MemoryAccess>> memrefToAccesses;
  SmallVector<TempProd> loadDataTemps;
  convertLoadsAndStores(body, memrefToAccesses, loadDataTemps, namer, builder);

  // maximize SSA, so that every value used in a basic block that does not
  // produce it is one of that basic block's arguments. this comes after the
  // conversion, so the new block arguments take handshake types, and start's
  // uses by the constants get block arguments too
  SSAMaximizationStrategy strategy;
  if (failed(maximizeSSA(body, strategy))) {
    return failure();
  }

  // collect all block uses that must be routed
  SetVector<BlockUse> blockUses = collectBlockUses(body);

  // cluster the block uses into sets, and we will run the
  // control-network routing algorithm per set
  // for now, we always do one set per unique block use
  // for maximum task-level parallelism
  // we will add support for varying this later
  SmallVector<SetVector<BlockUse>> blockUseSets;
  for (BlockUse blockUse : blockUses) {
    SetVector<BlockUse> blockUseSet;
    blockUseSet.insert(blockUse);
    blockUseSets.push_back(blockUseSet);
  }

  // build one cfg per set of block uses
  //
  // using small vector on the heap over std::vector
  // based on the programmers manual
  SmallVector<Cfg, 0> nonMemoryCfgs;
  for (const SetVector<BlockUse> &blockUseSet : blockUseSets) {
    nonMemoryCfgs.push_back(buildCfg(base, blockUseSet));
  }

  // for each cfg, optimize based on the block uses annotated on it
  for (Cfg &cfg : nonMemoryCfgs) {
    optimizeCfg(cfg);
  }
  // for each memory cfg, optimize
  for (auto &[key, memoryCfg] : memoryCfgs) {
    optimizeMemoryCfg(memoryCfg);
  }

  // run the control-network routing algorithm once per cfg
  MapVector<MemrefValue, MemoryInterfaceInputs> memoryInputs = routeCfgs(
      body, blockToCond, memStarts, nonMemoryCfgs, memoryCfgs, builder);

  // label every op with the number of its basic block (its bb attribute).
  // the memory interface builder reads it off each memory access
  // to group them, so it has to be set before the interfaces are built
  for (Block &block : body) {
    for (Operation &op : block) {
      setBB(&op, getBlockNumber(&block));
    }
  }

  // build the memory interfaces (MCs and LSQs),
  // one memory region (array) at a time

  // we need to store the end signals from each memory interface
  // to pass to the handshake end op
  SmallVector<Value> memEnds;

  // for each memory region (unpacking its memref and mem start function arg)
  for (auto [memref, memStart] : memStarts) {

    // what the region's memory routing delivers to its interfaces
    MemoryInterfaceInputs &inputs = memoryInputs[memref];

    // a memory region (array) without accesses
    // ends immediately after starting:
    // its memEnd comes from its memStart
    if (!inputs.ctrlEnd) {
      memEnds.push_back(memStart);
      continue;
    }

    // make the builder of MC and LSQ handshake operations
    MemoryInterfaceBuilder memBuilder(newFuncOp, memref, memStart,
                                      inputs.ctrlEnd, inputs.mcControls,
                                      inputs.lsqControls);

    // for each memory access of this memref
    // tell the builder about it
    for (MemoryAccess access : memrefToAccesses[memref]) {
      auto label = getDialectAttr<handshake::MemInterfaceAttr>(access);
      if (label.connectsToMC()) {
        // as either an MC memory access
        memBuilder.addMCPort(access);
      } else {
        // or an LSQ memory access
        memBuilder.addLSQPort(*label.getLsqGroup(), access);
      }
    }

    // instantiateInterfaces places either an MC op, an LSQ op, or both
    // and connects those ops to the handshake loads and stores
    // of that memref
    // it should just return the memEnd signal for us,
    // but it's used elsewhere, so not changing it for now
    handshake::MemoryControllerOp mcOp;
    handshake::LSQOp lsqOp;
    if (failed(memBuilder.instantiateInterfaces(builder, mcOp, lsqOp))) {
      return failure();
    }

    // take the end signal from the MC if it exists
    // or the LSQ if it does not
    memEnds.push_back(mcOp ? mcOp.getMemEnd() : lsqOp.getMemEnd());
  }

  for (auto &[memref, inputs] : memoryInputs) {
    bufferContinuingTokens(inputs.mcControls);
    bufferContinuingTokens(inputs.lsqControls);
  }

  // Since memBuilder.instantiateInterfaces rewires
  // the loads under the hood,
  // we need to just erase the temporary producers
  // rather than the typical resolution approach
  for (TempProd &dataTemp : loadDataTemps) {
    dataTemp.erase();
  }

  // Now we need to replace the func.return op
  // with a handshake.end op
  // This requires info from
  // before deleting the func.return op
  // and flattening the IR,
  // but the end cannot actually be created until after flattening

  // the end op's basic block is the exit basic block
  unsigned exitNumber = getBlockNumber(base.exit);

  // its operands are, in order-
  // the operands of the func.return op
  SmallVector<Value> endOperands(returnOp.getOperands());
  // the memEnd controls
  llvm::append_range(endOperands, memEnds);
  // and then the (fully redundant) end result
  // which is just start
  endOperands.push_back(start);

  // Erase the cf.br, cf.cond_br and func.return operations
  for (Block &block : body) {
    block.getTerminator()->erase();
  }

  // erase the BlockToCond temporaries,
  // now that the cf.cond_br ops are erased
  blockToCond.eraseTemps();

  // flatten the handshake function into a single "basic block"
  // (the entry basic block)
  // because our elastic routing is incompatible with block arguments
  // We do this by moving
  // all operations of a "basic block" to the entry block
  // And then erasing that block
  Block &entry = body.front();
  for (Block &block : llvm::make_early_inc_range(llvm::drop_begin(body))) {
    // After routing, the block args should have no uses
    for (BlockArgument arg : block.getArguments()) {
      assert(arg.use_empty() && "block argument still has a use");
    }

    // move the operations to the entry block
    entry.getOperations().splice(entry.end(), block.getOperations());
    // erase this block
    block.erase();
  }

  // actually create the end op
  builder.setInsertionPointToEnd(&entry);
  auto endOp = handshake::EndOp::create(builder, loc, endOperands);
  setBB(endOp, exitNumber);

  // delete the func.func op
  funcOp.erase();

  // and return
  return success();
}

struct CfToHandshakePass
    : public dynamatic::impl::CfToHandshakeBase<CfToHandshakePass> {
  using CfToHandshakeBase::CfToHandshakeBase;

  void runOnOperation() override {
    // get the operation
    ModuleOp modOp = getOperation();

    // use namer.isAnalysisValid to check the handshake.names are unique
    NameAnalysis &namer = getAnalysis<NameAnalysis>();
    if (!namer.isAnalysisValid())
      return signalPassFailure();

    // every load and store must be named, as their memory dependencies
    // refer to each other by name
    // And also nothing else should be named yet
    for (auto funcOp : modOp.getOps<func::FuncOp>()) {
      for (Operation &op : funcOp.getBody().getOps()) {
        bool isAccess = isa<memref::LoadOp, memref::StoreOp>(op);
        bool isNamed = namer.hasName(&op);
        if (isAccess && !isNamed) {
          op.emitError() << "memory access has no name";
          return signalPassFailure();
        }
        if (!isAccess && isNamed) {
          op.emitError() << "only memory accesses should be named";
          return signalPassFailure();
        }
      }
    }

    // handle the builder
    OpBuilder builder(&getContext());

    // a single function, with no calls, is supported for now
    SmallVector<func::FuncOp> funcOps;
    for (auto funcOp : modOp.getOps<func::FuncOp>()) {
      if (!funcOp.isExternal()) {
        funcOps.push_back(funcOp);
      }
    }
    if (funcOps.size() != 1) {
      modOp.emitError() << "cf to handshake supports a single function only";
      return signalPassFailure();
    }
    func::FuncOp funcOp = funcOps.front();
    for (auto callOp : funcOp.getBody().getOps<func::CallOp>()) {
      callOp.emitError() << "cf to handshake does not support function calls "
                            "yet";
      return signalPassFailure();
    }

    // Convert all CF internal array ops to RAM ops as step 1.
    // This is helpful for internal array processing,
    // as it makes the memrefs easy to process
    if (failed(convertInternalArrays(funcOp, namer, builder))) {
      return signalPassFailure();
    }

    // remove global ops
    // (both internal arrays and external functions have been processed)
    for (auto globalOp :
         llvm::make_early_inc_range(modOp.getOps<memref::GlobalOp>())) {
      globalOp.erase();
    }

    // run cf to handshake on the one function
    if (failed(lowerFunction(funcOp, namer, builder))) {
      return signalPassFailure();
    }

    // name all unnamed operations in the IR
    if (failed(namer.walk(NameAnalysis::UnnamedBehavior::NAME)))
      return signalPassFailure();
    markAnalysesPreserved<NameAnalysis>();
  }
};
} // namespace
