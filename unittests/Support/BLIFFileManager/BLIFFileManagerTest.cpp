//===- BLIFFileManagerTest.cpp - Unit tests for BLIFFileManager -*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/BLIFFileManager.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/InitAllDialects.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/APInt.h"
#include <filesystem>
#include <gtest/gtest.h>
#include <string>

#ifndef BLIF_TEST_DIR
#error                                                                         \
    "BLIF_TEST_DIR must be defined as a compiler definition pointing to data/aig"
#endif
#ifndef DYNAMATIC_ROOT
#error                                                                         \
    "DYNAMATIC_ROOT must be defined as a compiler definition pointing to the Dynamatic root directory"
#endif
#ifndef RTL_JSON_FILE
#error                                                                         \
    "RTL_JSON_FILE must be defined as a compiler definition pointing to the RTL JSON configuration file"
#endif

using namespace dynamatic;
using namespace dynamatic::handshake;
namespace fs = std::filesystem;

class BLIFFileManagerTest : public ::testing::Test {
protected:
  mlir::MLIRContext ctx;
  BLIFFileManager manager;

  BLIFFileManagerTest()
      : manager(BLIF_TEST_DIR, DYNAMATIC_ROOT, RTL_JSON_FILE) {}

  void SetUp() override {
    mlir::DialectRegistry registry;
    dynamatic::registerAllDialects(registry);
    ctx.appendDialectRegistry(registry);
    ctx.loadAllAvailableDialects();
  }

  // Helper function to create a control argument at a block and return it.
  static mlir::Value addCtrlArg(mlir::Block &b, mlir::MLIRContext &ctx) {
    return b.addArgument(ControlType::get(&ctx), mlir::UnknownLoc::get(&ctx));
  }
  // Helper function to create a channel argument at a block and return it.
  static mlir::Value addChannelArg(mlir::Block &b, mlir::MLIRContext &ctx,
                                   unsigned width) {
    return b.addArgument(ChannelType::get(mlir::IntegerType::get(&ctx, width)),
                         mlir::UnknownLoc::get(&ctx));
  }
  // Function to run the BLIFFileManager on an operation
  void checkOp(mlir::Operation *op, const char *expectedSubstr) {
    // Run the BLIFFileManager to get the BLIF file path for the operation, and
    // check that it contains the expected substring (derived from the operation
    // name and parameters).
    std::string path = manager.getBlifFilePathForHandshakeOp(op);
    EXPECT_NE(path.find(expectedSubstr), std::string::npos)
        << "[" << expectedSubstr << "] unexpected path: " << path;
    // Check that the file exists at the generated path.
    EXPECT_TRUE(fs::exists(path))
        << "[" << expectedSubstr << "] BLIF file not found: " << path;
  }

  // Checks all two-input arith integer ops with no additional attribute
  template <typename OpTy>
  void checkBinaryIntOp(const char *expectedSubstr, unsigned width = 32) {
    // Create a block and add two channel arguments of the specified width.
    mlir::Block block;
    mlir::OpBuilder b(&ctx);
    auto lhs = addChannelArg(block, ctx, width),
         rhs = addChannelArg(block, ctx, width);
    b.setInsertionPointToEnd(&block);
    checkOp(b.create<OpTy>(b.getUnknownLoc(), lhs, rhs), expectedSubstr);
  }
};

// Path convention (driven by getRTLParameters() order):
//   addi/subi/andi/ori/xori/shli/shrsi/shrui  -> <op>/BITWIDTH/<op>.blif
//   muli ->    muli/BITWIDTH/LATENCY/muli.blif
//   cmpi -> cmpi/PREDICATE/BITWIDTH/cmpi.blif
//   constant -> constant/VALUE/BITWIDTH/constant.blif
//   select -> select/BITWIDTH/select.blif

TEST_F(BLIFFileManagerTest, AllSupportedOpsPathAndFileExist) {
  // Test with bitwidth 32
  checkBinaryIntOp<handshake::AddIOp>("addi/32");
  checkBinaryIntOp<handshake::SubIOp>("subi/32");
  checkBinaryIntOp<handshake::AndIOp>("andi/32");
  checkBinaryIntOp<handshake::OrIOp>("ori/32");
  checkBinaryIntOp<handshake::XOrIOp>("xori/32");
  checkBinaryIntOp<handshake::ShLIOp>("shli/32");
  checkBinaryIntOp<handshake::ShRSIOp>("shrsi/32");
  checkBinaryIntOp<handshake::ShRUIOp>("shrui/32");

  // muli test with latency 4 and bitwidth 32
  {
    mlir::Block block;
    mlir::OpBuilder b(&ctx);
    auto lhs = addChannelArg(block, ctx, 32),
         rhs = addChannelArg(block, ctx, 32);
    b.setInsertionPointToEnd(&block);
    auto op = b.create<handshake::MulIOp>(b.getUnknownLoc(), lhs, rhs);
    op->setAttr("latency",
                mlir::IntegerAttr::get(mlir::IntegerType::get(&ctx, 64), 4));
    checkOp(op, "muli/32/4");
  }

  // cmpi test with predicate "slt" and bitwidth 32
  {
    mlir::Block block;
    mlir::OpBuilder b(&ctx);
    auto lhs = addChannelArg(block, ctx, 32),
         rhs = addChannelArg(block, ctx, 32);
    b.setInsertionPointToEnd(&block);
    checkOp(b.create<handshake::CmpIOp>(b.getUnknownLoc(), CmpIPredicate::slt,
                                        lhs, rhs),
            "cmpi/slt/32");
  }

  // constant test with value -25 (1111100111 in binary) and bitwidth 10
  {
    mlir::Block block;
    mlir::OpBuilder b(&ctx);
    auto ctrl = addCtrlArg(block, ctx);
    b.setInsertionPointToEnd(&block);
    auto valAttr = mlir::IntegerAttr::get(mlir::IntegerType::get(&ctx, 10),
                                          llvm::APInt(10, -25, true));
    checkOp(b.create<handshake::ConstantOp>(b.getUnknownLoc(), valAttr, ctrl),
            "constant/1111100111/10");
  }

  // select test with bitwidth 32
  {
    mlir::Block block;
    mlir::OpBuilder b(&ctx);
    auto cond = addChannelArg(block, ctx, 1);
    auto tval = addChannelArg(block, ctx, 32);
    auto fval = addChannelArg(block, ctx, 32);
    b.setInsertionPointToEnd(&block);
    checkOp(b.create<handshake::SelectOp>(b.getUnknownLoc(), cond, tval, fval),
            "select/32");
  }
}
