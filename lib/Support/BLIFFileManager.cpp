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

namespace dynamatic {

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
      llvm::errs() << "BLIF file for operation `" << getUniqueName(op)
                   << "` does not exist: " << blifFileName
                   << ". Check the file path specified for this operation is "
                      "correct. If it is not present, you should generate it "
                      "using the blif script generator.\n";
      assert(false && "Check the file path specified for this operation is "
                      "built correctly");
    }
  }
  return blifFileName;
}

// Formats a bit-indexed port name: "sig[bit]".
// When width == 1 the name is returned unchanged (no "[0]" suffix).
// If baseName already ends with "[N]" the indices are linearised.
std::string legalizeDataPortName(StringRef baseName, unsigned bit,
                                 unsigned width) {
  if (width == 1)
    return baseName.str();
  // Reuse the formatArrayName() helper from Step 1 / Layer 2.
  return formatArrayName(baseName.str(), bit, width);
}

// Inserts a suffix string before the first '[' in the input name, or appends
// it at the end if no '[' is found.
// Example: legalizeControlPortName("data[3]", "_valid") -> "data_valid[3]"
std::string legalizeControlPortName(const std::string &name,
                                    const std::string &suffix) {
  std::size_t pos = name.find('[');
  if (pos != std::string::npos) {
    std::string result = name;
    result.insert(pos, suffix);
    return result;
  }
  return name + suffix;
}

// Converts a (root, index) pair into the canonical "root[index]" form.
// If root already contains "[N]", the old index is linearised with
//  arrayWidth before adding index.
// Example: formatArrayName("data[2]", 3, 4) becomes "data[11]"  (2*4 + 3)
std::string formatArrayName(const std::string &root, unsigned index,
                            unsigned arrayWidth) {
  static const std::regex arrayPattern(R"((\w+)\[(\d+)\])");
  std::smatch m;
  if (std::regex_match(root, m, arrayPattern)) {
    assert(arrayWidth != 0 && "arrayWidth required for already-indexed names");
    unsigned linearised = std::stoi(m[2].str()) * arrayWidth + index;
    return m[1].str() + "[" + std::to_string(linearised) + "]";
  }
  return root + "[" + std::to_string(index) + "]";
}

// Legalizes a list of handshake port names by converting the "root_N" index
// pattern (used internally by NamedIOInterface) into the "root[N]" array
// notation expected by the BLIF importer.
// Example: ["data_0", "data_1", "valid"] -> ["data[0]", "data[1]", "valid"]
void legalizeBlifPortNames(SmallVector<std::string> &names) {
  static const std::regex indexPat(R"((\w+)_(\d+))");
  SmallVector<std::string> result;
  result.reserve(names.size());

  for (auto &name : names) {
    std::smatch m;
    if (!std::regex_match(name, m, indexPat)) {
      result.push_back(name);
      continue;
    }
    std::string root = m[1].str();
    unsigned idx = std::stoi(m[2].str());
    if (idx == 0) {
      result.push_back(root);
    } else {
      if (idx == 1) {
        // Back-patch the root entry (index 0) to have the "[0]" suffix,
        // since it was originally stored without an index.
        auto *it = std::find(result.begin(), result.end(), root);
        assert(it != result.end() && "index-0 port not found for back-patch");
        *it = formatArrayName(root, 0);
      }
      result.push_back(formatArrayName(root, idx));
    }
  }
  names = std::move(result);
}

} // namespace dynamatic