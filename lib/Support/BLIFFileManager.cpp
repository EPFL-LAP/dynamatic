//===- BLIFFileManager.cpp - Support functions for BLIF importer -*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the function to retrieve the path of the blif file for a
// given handshake operation. The path is created by combining the base path
// contining all the blif files and the operation name and parameters. The file
// is expected to be named in the format <op_name>_<param1>_<param2>_..._.blif.
// If the file does not exist, an error is emitted.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/BLIFFileManager.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

// Function to combine parameter values, module type and blif directory path
// to create the blif file path
std::string
BLIFFileManager::combineBlifFilePath(std::string moduleType,
                                     std::vector<std::string> paramValues,
                                     std::string extraSuffix) {
  std::string blifFilePath = blifDirPath + "/" + moduleType + extraSuffix;
  for (const auto &paramValue : paramValues) {
    blifFilePath += "/" + paramValue;
  }
  blifFilePath += "/" + moduleType + extraSuffix + ".blif";
  return blifFilePath;
}

// Function to retrieve the path of the blif file for a given handshake
// operation
std::string BLIFFileManager::getBlifFilePathForHandshakeOp(Operation *op) {
  // Depending on the operation type, there is a different number and type of
  // parameters. For now, we use a switch hard-coded on the operation type. In
  // the future, this should be improved by allowing extraction of parameter
  // information from the operation itself
  std::string moduleType = op->getName().getStringRef().str();
  // Erase the dialect name from the moduleType
  moduleType = moduleType.substr(moduleType.find('.') + 1);
  std::string blifFileName;
  llvm::TypeSwitch<Operation *>(op)
      .Case<handshake::AddIOp, handshake::AndIOp, handshake::CmpIOp,
            handshake::OrIOp, handshake::ShLIOp, handshake::ShRSIOp,
            handshake::ShRUIOp, handshake::SubIOp, handshake::XOrIOp,
            handshake::MulIOp, handshake::DivSIOp, handshake::DivUIOp,
            handshake::SelectOp, handshake::SIToFPOp, handshake::FPToSIOp,
            handshake::ExtFOp, handshake::TruncFOp>([&](auto) {
        unsigned dataWidth =
            handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());
        blifFileName =
            combineBlifFilePath(moduleType, {std::to_string(dataWidth)});
      })
      .Case<handshake::ConstantOp>([&](handshake::ConstantOp constOp) {
        handshake::ChannelType cstType = constOp.getResult().getType();
        // Get the data width of the constant operation
        unsigned dataWidth = cstType.getDataBitWidth();
        blifFileName =
            combineBlifFilePath(moduleType, {std::to_string(dataWidth)});
      })
      .Case<handshake::BranchOp, handshake::SinkOp>([&](auto) {
        unsigned dataWidth =
            handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());
        if (dataWidth == 0) {
          blifFileName = combineBlifFilePath(moduleType, {}, "_dataless");
        } else {
          blifFileName =
              combineBlifFilePath(moduleType, {std::to_string(dataWidth)});
        }
      })
      .Case<handshake::BufferOp>([&](handshake::BufferOp op) {
        std::string bufferType =
            handshake::stringifyEnum(op.getBufferType()).str();
        unsigned dataWidth =
            handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());
        if (dataWidth == 0) {
          blifFileName = combineBlifFilePath(bufferType, {}, "_dataless");
        } else {
          blifFileName =
              combineBlifFilePath(bufferType, {std::to_string(dataWidth)});
        }
      })
      .Case<handshake::ConditionalBranchOp>(
          [&](handshake::ConditionalBranchOp cbrOp) {
            unsigned dataWidth = handshake::getHandshakeTypeBitWidth(
                cbrOp.getDataOperand().getType());
            if (dataWidth == 0) {
              blifFileName = combineBlifFilePath(moduleType, {}, "_dataless");
            } else {
              blifFileName =
                  combineBlifFilePath(moduleType, {std::to_string(dataWidth)});
            }
          })
      .Case<handshake::ControlMergeOp>([&](handshake::ControlMergeOp cmergeOp) {
        unsigned size = cmergeOp.getDataOperands().size();
        unsigned dataWidth =
            handshake::getHandshakeTypeBitWidth(cmergeOp.getResult().getType());
        unsigned indexType =
            handshake::getHandshakeTypeBitWidth(cmergeOp.getIndex().getType());
        if (dataWidth == 0) {
          blifFileName = combineBlifFilePath(
              moduleType, {std::to_string(size), std::to_string(indexType)},
              "_dataless");
        } else {
          assert(false && "ControlMerge with data not supported yet");
        }
      })
      .Case<handshake::ExtSIOp, handshake::ExtUIOp, handshake::ExtFOp,
            handshake::TruncIOp, handshake::TruncFOp>([&](auto op) {
        unsigned inputWidth =
            handshake::getHandshakeTypeBitWidth(op.getOperand().getType());
        unsigned outputWidth =
            handshake::getHandshakeTypeBitWidth(op.getResult().getType());
        blifFileName =
            combineBlifFilePath(moduleType, {std::to_string(inputWidth),
                                             std::to_string(outputWidth)});
      })
      .Case<handshake::ForkOp, handshake::LazyForkOp>([&](auto) {
        unsigned size = op->getNumResults();
        unsigned dataWidth =
            handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());
        if (dataWidth == 0) {
          blifFileName = combineBlifFilePath(moduleType, {std::to_string(size)},
                                             "_dataless");
        } else {
          blifFileName = combineBlifFilePath(
              moduleType, {std::to_string(size), std::to_string(dataWidth)},
              "_type");
        }
      })
      .Case<handshake::MuxOp>([&](handshake::MuxOp muxOp) {
        unsigned size = muxOp.getDataOperands().size();
        unsigned dataWidth =
            handshake::getHandshakeTypeBitWidth(muxOp.getResult().getType());
        unsigned selectType = handshake::getHandshakeTypeBitWidth(
            muxOp.getSelectOperand().getType());
        blifFileName = combineBlifFilePath(
            moduleType, {std::to_string(size), std::to_string(dataWidth),
                         std::to_string(selectType)});
      })
      .Case<handshake::MergeOp>([&](handshake::MergeOp mergeOp) {
        unsigned size = mergeOp.getDataOperands().size();
        unsigned dataWidth = handshake::getHandshakeTypeBitWidth(
            mergeOp.getDataOperands()[0].getType());
        blifFileName = combineBlifFilePath(
            moduleType, {std::to_string(size), std::to_string(dataWidth)});
      })
      .Case<handshake::LoadOp, handshake::StoreOp>([&](auto op) {
        unsigned dataWidth =
            handshake::getHandshakeTypeBitWidth(op.getDataInput().getType());
        unsigned addrType =
            handshake::getHandshakeTypeBitWidth(op.getAddressInput().getType());
        blifFileName = combineBlifFilePath(
            moduleType, {std::to_string(dataWidth), std::to_string(addrType)});
      })
      .Case<handshake::SourceOp>(
          [&](auto) { blifFileName = combineBlifFilePath(moduleType, {}); })
      .Default([&](auto) { blifFileName = ""; });
  // Check blifFileName path exists
  if (blifFileName != "") {
    // Check if the file exists
    if (!std::filesystem::exists(blifFileName)) {
      llvm::errs() << "BLIF file for operation " << getUniqueName(op)
                   << "does not exist: " << blifFileName << "\n";
      assert(false && "Check the file path specified for this operation is "
                      "built correctly");
    }
  }
  return blifFileName;
}
