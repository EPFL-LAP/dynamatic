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
// If the file does not exist, an attempt is made to generate it using
// BLIFGenerator (requires Yosys and ABC to be configured at build time).
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/BLIFFileManager.h"
#include "dynamatic/Support/BLIFGenerator.h"
#include "llvm/Support/raw_ostream.h"

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
  std::string
      genComponent; // full component name looked up in the JSON config (=
                    // moduleType + extraSuffix, e.g. "fork_dataless").
  std::vector<std::string>
      genParams; // ordered parameter values matching the JSON generic params.

  llvm::TypeSwitch<Operation *>(op)
      .Case<handshake::AddIOp, handshake::AndIOp, handshake::OrIOp,
            handshake::ShLIOp, handshake::ShRSIOp, handshake::ShRUIOp,
            handshake::SubIOp, handshake::XOrIOp, handshake::MulIOp,
            handshake::DivSIOp, handshake::DivUIOp, handshake::SelectOp,
            handshake::SIToFPOp, handshake::FPToSIOp>([&](auto) {
        unsigned dataWidth =
            handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());
        std::string dw = std::to_string(dataWidth);
        blifFileName = combineBlifFilePath(moduleType, {dw});
        genComponent = moduleType;
        genParams = {dw};
      })
      .Case<handshake::CmpIOp>([&](handshake::CmpIOp cmpOp) {
        unsigned dataWidth =
            handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());
        std::string dw = std::to_string(dataWidth);
        std::string predStr = stringifyEnum(cmpOp.getPredicate()).str();
        blifFileName = combineBlifFilePath(moduleType, {dw, predStr});
        genComponent = moduleType;
        genParams = {dw, predStr};
      })
      .Case<handshake::ConstantOp>([&](handshake::ConstantOp constOp) {
        handshake::ChannelType cstType = constOp.getResult().getType();
        unsigned dataWidth = cstType.getDataBitWidth();
        std::string dw = std::to_string(dataWidth);
        // Build a binary string of the constant value (MSB first, dataWidth
        // characters) so that different values get distinct BLIF files and
        // the correct value is passed to the Verilog generator.
        auto valueAttr = constOp->getAttrOfType<mlir::IntegerAttr>("value");
        llvm::APInt apVal = valueAttr.getValue();
        std::string binStr(dataWidth, '0');
        for (unsigned i = 0; i < dataWidth; ++i)
          binStr[dataWidth - 1 - i] = apVal[i] ? '1' : '0';
        blifFileName = combineBlifFilePath(moduleType, {dw, binStr});
        genComponent = moduleType;
        genParams = {dw, binStr};
      })
      .Case<handshake::BranchOp, handshake::SinkOp>([&](auto) {
        unsigned dataWidth =
            handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());
        if (dataWidth == 0) {
          blifFileName = combineBlifFilePath(moduleType, {}, "_dataless");
          genComponent = moduleType + "_dataless";
          genParams = {};
        } else {
          std::string dw = std::to_string(dataWidth);
          blifFileName = combineBlifFilePath(moduleType, {dw});
          genComponent = moduleType;
          genParams = {dw};
        }
      })
      .Case<handshake::BufferOp>([&](handshake::BufferOp bufOp) {
        unsigned dataWidth =
            handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());
        unsigned numSlots = static_cast<unsigned>(bufOp.getNumSlots());
        bool isDataless = (dataWidth == 0);
        std::string dw = std::to_string(dataWidth);
        std::string ns = std::to_string(numSlots);

        // Map BufferType enum to the JSON module-name base and whether
        // NUM_SLOTS is a range parameter in the BLIF path.
        std::string base;
        bool hasNumSlots = false;
        switch (bufOp.getBufferType()) {
        case handshake::BufferType::ONE_SLOT_BREAK_DV:
          base = "oehb";
          break;
        case handshake::BufferType::ONE_SLOT_BREAK_R:
          base = "tehb";
          break;
        case handshake::BufferType::FIFO_BREAK_NONE:
          base = "tfifo";
          hasNumSlots = true;
          break;
        case handshake::BufferType::ONE_SLOT_BREAK_DVR:
          base = "one_slot_break_dvr";
          break;
        case handshake::BufferType::SHIFT_REG_BREAK_DV:
          base = "shift_reg_break_dv";
          hasNumSlots = true;
          break;
        default:
          blifFileName = "";
          return;
        }

        std::string component = base + (isDataless ? "_dataless" : "");
        std::vector<std::string> params;
        if (hasNumSlots)
          params.push_back(ns);
        if (!isDataless)
          params.push_back(dw);

        blifFileName = combineBlifFilePath(component, params);
        genComponent = component;
        genParams = params;
      })
      .Case<handshake::ConditionalBranchOp>(
          [&](handshake::ConditionalBranchOp cbrOp) {
            unsigned dataWidth = handshake::getHandshakeTypeBitWidth(
                cbrOp.getDataOperand().getType());
            if (dataWidth == 0) {
              blifFileName = combineBlifFilePath(moduleType, {}, "_dataless");
              genComponent = moduleType + "_dataless";
              genParams = {};
            } else {
              std::string dw = std::to_string(dataWidth);
              blifFileName = combineBlifFilePath(moduleType, {dw});
              genComponent = moduleType;
              genParams = {dw};
            }
          })
      .Case<handshake::ControlMergeOp>([&](handshake::ControlMergeOp cmergeOp) {
        unsigned size = cmergeOp.getDataOperands().size();
        unsigned dataWidth =
            handshake::getHandshakeTypeBitWidth(cmergeOp.getResult().getType());
        unsigned indexType =
            handshake::getHandshakeTypeBitWidth(cmergeOp.getIndex().getType());
        if (dataWidth == 0) {
          std::string sz = std::to_string(size);
          std::string it = std::to_string(indexType);
          blifFileName = combineBlifFilePath(moduleType, {sz, it}, "_dataless");
          genComponent = moduleType + "_dataless";
          genParams = {sz, it};
        } else {
          assert(false && "ControlMerge with data not supported yet");
        }
      })
      .Case<handshake::ExtSIOp, handshake::ExtUIOp, handshake::ExtFOp,
            handshake::TruncIOp, handshake::TruncFOp>([&](auto extOp) {
        unsigned inputWidth =
            handshake::getHandshakeTypeBitWidth(extOp.getOperand().getType());
        unsigned outputWidth =
            handshake::getHandshakeTypeBitWidth(extOp.getResult().getType());
        std::string iw = std::to_string(inputWidth);
        std::string ow = std::to_string(outputWidth);
        blifFileName = combineBlifFilePath(moduleType, {iw, ow});
        genComponent = moduleType;
        genParams = {iw, ow};
      })
      .Case<handshake::ForkOp, handshake::LazyForkOp>([&](auto) {
        unsigned size = op->getNumResults();
        unsigned dataWidth =
            handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());
        std::string sz = std::to_string(size);
        if (dataWidth == 0) {
          blifFileName = combineBlifFilePath(moduleType, {sz}, "_dataless");
          genComponent = moduleType + "_dataless";
          genParams = {sz};
        } else {
          std::string dw = std::to_string(dataWidth);
          blifFileName = combineBlifFilePath(moduleType, {sz, dw}, "_type");
          genComponent = moduleType + "_type";
          genParams = {sz, dw};
        }
      })
      .Case<handshake::MuxOp>([&](handshake::MuxOp muxOp) {
        unsigned size = muxOp.getDataOperands().size();
        unsigned dataWidth =
            handshake::getHandshakeTypeBitWidth(muxOp.getResult().getType());
        unsigned selectType = handshake::getHandshakeTypeBitWidth(
            muxOp.getSelectOperand().getType());
        std::string sz = std::to_string(size);
        std::string dw = std::to_string(dataWidth);
        std::string st = std::to_string(selectType);
        blifFileName = combineBlifFilePath(moduleType, {sz, dw, st});
        genComponent = moduleType;
        genParams = {sz, dw, st};
      })
      .Case<handshake::MergeOp>([&](handshake::MergeOp mergeOp) {
        unsigned size = mergeOp.getDataOperands().size();
        unsigned dataWidth = handshake::getHandshakeTypeBitWidth(
            mergeOp.getDataOperands()[0].getType());
        std::string sz = std::to_string(size);
        std::string dw = std::to_string(dataWidth);
        // If dataWidth is 0, we use a special "_dataless" suffix
        if (dataWidth == 0) {
          blifFileName = combineBlifFilePath(moduleType, {sz}, "_dataless");
          genComponent = moduleType + "_dataless";
          genParams = {sz};
        } else {
          // If not, we generate the regular merge
          blifFileName = combineBlifFilePath(moduleType, {sz, dw});
          genComponent = moduleType;
          genParams = {sz, dw};
        }
      })
      .Case<handshake::LoadOp, handshake::StoreOp>([&](auto lsOp) {
        unsigned dataWidth =
            handshake::getHandshakeTypeBitWidth(lsOp.getDataInput().getType());
        unsigned addrType = handshake::getHandshakeTypeBitWidth(
            lsOp.getAddressInput().getType());
        std::string dw = std::to_string(dataWidth);
        std::string at = std::to_string(addrType);
        blifFileName = combineBlifFilePath(moduleType, {dw, at});
        genComponent = moduleType;
        genParams = {dw, at};
      })
      .Case<handshake::SourceOp>([&](auto) {
        blifFileName = combineBlifFilePath(moduleType, {});
        genComponent = moduleType;
        genParams = {};
      })
      .Case<handshake::EndOp>([&](auto) {
        // EndOp is a function terminator with no hardware implementation.
        blifFileName = "";
      })
      .Case<handshake::MemoryControllerOp>([&](handshake::MemoryControllerOp
                                                   mcOp) {
        assert(
            false &&
            "MemoryController is not currently handled since "
            "it has one side of the ports in handshake and the "
            "other side as non-handshake. This requires a more complex "
            "HandshakeToSynth conversion that is not currently implemented.");
        MCPorts ports = mcOp.getPorts();
        unsigned numControls = ports.getNumPorts<ControlPort>();
        unsigned numLoads = ports.getNumPorts<LoadPort>();
        unsigned numStores = ports.getNumPorts<StorePort>();
        std::string nc = std::to_string(numControls);
        std::string nl = std::to_string(numLoads);
        std::string ns = std::to_string(numStores);
        std::string dw = std::to_string(ports.dataWidth);
        std::string aw = std::to_string(ports.addrWidth);

        if (numControls == 0 && numStores == 0) {
          genComponent = "mem_controller_storeless";
          genParams = {nl, dw, aw};
        } else if (numLoads == 0) {
          genComponent = "mem_controller_loadless";
          genParams = {nc, ns, dw, aw};
        } else {
          genComponent = "mem_controller";
          genParams = {nc, nl, ns, dw, aw};
        }
        blifFileName = combineBlifFilePath(genComponent, genParams);
      })
      .Default([&](auto) { blifFileName = ""; });

  if (blifFileName.empty())
    return blifFileName;

  // Check if the file exists; if not, attempt to generate it.
  if (!std::filesystem::exists(blifFileName)) {
#if defined(DYNAMATIC_YOSYS_EXECUTABLE) && defined(DYNAMATIC_ABC_EXECUTABLE)
    llvm::errs() << "[BLIFFileManager] BLIF file missing, generating: "
                 << blifFileName << "\n";
    BLIFGenerator gen(blifDirPath, DYNAMATIC_YOSYS_EXECUTABLE,
                      DYNAMATIC_ABC_EXECUTABLE);
    bool ok = gen.generate(genComponent, genParams);
    if (!ok || !std::filesystem::exists(blifFileName)) {
      llvm::errs() << "Failed to generate BLIF file for operation `"
                   << getUniqueName(op) << "`: " << blifFileName
                   << ". Ensure that Yosys and ABC are correctly configured.\n";
      assert(false && "BLIF generation failed");
    }
#else
    llvm::errs() << "BLIF file for operation `" << getUniqueName(op)
                 << "` does not exist: " << blifFileName
                 << ". Check the file path specified for this operation is "
                    "correct. If it is not present, you should generate it "
                    "using the blif script generator.\n";
    assert(false && "Check the file path specified for this operation is "
                    "built correctly");
#endif
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