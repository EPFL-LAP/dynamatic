//===- PropertyExprTest.cpp - Tests of formal property expressions --------===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests the SMV printing and the JSON serialization of the expressions of
// formal properties.
//
//===----------------------------------------------------------------------===//

#include "experimental/Support/PropertyExpr.h"
#include "dynamatic/Dialect/Handshake/HandshakeOpInternalStateNamer.h"
#include "experimental/Support/FormalProperty.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <vector>

using namespace dynamatic;
using namespace dynamatic::handshake;

/// The sent state of output `outs_<index>` of the 1-bit fork `fork0`.
static PropertyExpr sent(unsigned index) {
  return State(EagerForkSentNamer(
      /*opName=*/"fork0", /*channelName=*/"outs_" + std::to_string(index),
      /*channelSize=*/1));
}

/// The valid signal of the input of `fork0`.
static Signal insValid() {
  return Signal(Signal::Valid, "fork0", Signal::Operand, /*portIndex=*/0,
                "ins");
}

/// The valid signal of output `outs_<index>` of `fork0`.
static Signal outsValid(unsigned index) {
  return Signal(Signal::Valid, "fork0", Signal::Result, /*portIndex=*/index,
                "outs_" + std::to_string(index));
}

/// Serializes `expr` to JSON text and parses it back.
static PropertyExpr roundTrip(const PropertyExpr &expr) {
  std::string text = llvm::formatv("{0}", toJSON(expr)).str();
  llvm::Expected<llvm::json::Value> value = llvm::json::parse(text);
  EXPECT_TRUE(bool(value));
  llvm::json::Path::Root root;
  PropertyExpr parsed;
  EXPECT_TRUE(fromJSON(*value, parsed, llvm::json::Path(root)));
  return parsed;
}

TEST(PropertyExprTest, PrintsLeaves) {
  EXPECT_EQ(sent(0).toSMV(), "fork0.outs_0_sent");
  EXPECT_EQ(PropertyExpr(insValid()).toSMV(), "fork0.ins_valid");
  EXPECT_EQ(PropertyExpr(Signal(Signal::Ready, "fork0", Signal::Operand,
                                /*portIndex=*/0, "ins"))
                .toSMV(),
            "fork0.ins_ready");
  EXPECT_EQ(PropertyExpr(Constant::integer(-3)).toSMV(), "-3");
  EXPECT_EQ(PropertyExpr(Constant::boolean(true)).toSMV(), "TRUE");
}

TEST(PropertyExprTest, ParenthesizesEveryOperation) {
  // count(sents) > 0 & ins_valid -> FALSE
  PropertyExpr anySent =
      BinaryExpr(NaryExpr(NaryExpr::Count, {sent(0), sent(1)}), BinaryExpr::Gt,
                 Constant::integer(0));
  PropertyExpr expr =
      implies(conjunction({anySent, insValid()}), disjunction({}));
  EXPECT_EQ(expr.toSMV(), "(((count(fork0.outs_0_sent, fork0.outs_1_sent) > 0) "
                          "& fork0.ins_valid) -> FALSE)");
}

TEST(PropertyExprTest, ParenthesizesConstrainedStates) {
  // The constrained name is a conjunction, which the negation must enclose
  PropertyExpr constrained =
      State(EagerForkSentNamer("fork0", "outs_0", 1), /*constraint=*/1);
  EXPECT_EQ((!constrained).toSMV(),
            "(! (fork0.outs_0_sent & (fork0.ins = TRUE)))");
}

TEST(PropertyExprTest, CollapsesSingleOperands) {
  EXPECT_EQ(conjunction({sent(0)}).toSMV(), "fork0.outs_0_sent");
  EXPECT_EQ(conjunction({}).toSMV(), "TRUE");
  EXPECT_EQ(sum({}).toSMV(), "0");
}

TEST(PropertyExprTest, DetectsTemporalOperators) {
  PropertyExpr invariant = implies(sent(0), sent(1));
  EXPECT_FALSE(invariant.isTemporal());
  PropertyExpr eventually =
      UnaryExpr(UnaryExpr::AF, UnaryExpr(UnaryExpr::AG, !sent(0)));
  EXPECT_TRUE(eventually.isTemporal());
  EXPECT_EQ(eventually.toSMV(), "(AF (AG (! fork0.outs_0_sent)))");
  // Nested below a boolean operator
  EXPECT_TRUE(conjunction({invariant, eventually}).isTemporal());
}

TEST(PropertyExprTest, RoundTripsThroughJSON) {
  PropertyExpr constrainedSent =
      State(EagerForkSentNamer("fork0", "outs_0", 1), /*constraint=*/0);
  PropertyExpr flow =
      BinaryExpr(sum({BinaryExpr(UnaryExpr(UnaryExpr::ToInt, constrainedSent),
                                 BinaryExpr::Mul, Constant::integer(-1)),
                      UnaryExpr(UnaryExpr::ToInt, sent(1))}),
                 BinaryExpr::Eq, Constant::integer(0));
  PropertyExpr temporal =
      UnaryExpr(UnaryExpr::EF, BinaryExpr(insValid(), BinaryExpr::Iff,
                                          Constant::boolean(false)));
  for (const PropertyExpr &expr : {flow, temporal})
    EXPECT_EQ(roundTrip(expr).toSMV(), expr.toSMV());
}

TEST(PropertyExprTest, RejectsMalformedJSON) {
  llvm::json::Path::Root root;
  PropertyExpr parsed;
  llvm::json::Value unknownOp = llvm::json::Object{{"op", "xor"}};
  EXPECT_FALSE(fromJSON(unknownOp, parsed, llvm::json::Path(root)));
  // A negation takes exactly one operand
  llvm::json::Value wrongArity = llvm::json::Object{
      {"op", "not"},
      {"operands", llvm::json::Array{toJSON(sent(0)), toJSON(sent(1))}}};
  EXPECT_FALSE(fromJSON(wrongArity, parsed, llvm::json::Path(root)));
}

TEST(FormalPropertyTest, RoundTripsThroughJSON) {
  FormalProperty property{
      /*id=*/7,
      /*kind=*/FormalProperty::VALID_EQUIVALENCE.str(),
      /*tag=*/FormalProperty::TAG::OPT,
      /*check=*/true,
      /*expr=*/
      BinaryExpr(outsValid(0), BinaryExpr::Iff, outsValid(1))};
  llvm::json::Path::Root root;
  FormalProperty parsed;
  ASSERT_TRUE(fromJSON(toJSON(property), parsed, llvm::json::Path(root)));
  EXPECT_EQ(parsed.id, property.id);
  EXPECT_EQ(parsed.kind, property.kind);
  EXPECT_EQ(parsed.tag, property.tag);
  EXPECT_EQ(parsed.check, property.check);
  EXPECT_EQ(parsed.expr.toSMV(), property.expr.toSMV());
}

TEST(FormalPropertyTest, FindsTheChannelsOfValidSignals) {
  // Absence of backpressure on outs_1, whose user is the input of fork1: only
  // the valid signal of the result port identifies a channel
  FormalProperty property{
      /*id=*/0,
      /*kind=*/FormalProperty::ABSENCE_OF_BACKPRESSURE.str(),
      /*tag=*/FormalProperty::TAG::OPT,
      /*check=*/std::nullopt,
      /*expr=*/
      implies(outsValid(1), Signal(Signal::Ready, "fork1", Signal::Operand,
                                   /*portIndex=*/0, "ins"))};
  std::vector<Signal> valids = property.getResultValidSignals();
  ASSERT_EQ(valids.size(), 1u);
  EXPECT_EQ(valids[0].getOpName(), "fork0");
  EXPECT_EQ(valids[0].getPortIndex(), 1u);
}
