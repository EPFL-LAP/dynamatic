//===- export-vhdl.cpp - Export VHDL from netlist-level IR ------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Experimental tool that exports nuXmv-compatible format from a netlist-level
// IR expressed in the handshake dialect
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/TimingModels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include <cstddef>
#include <string>
#include <utility>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;

static cl::OptionCategory mainCategory("Application options");

static cl::opt<std::string> inputFileName(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::cat(mainCategory));

static cl::opt<std::string> timingDBFilepath(
    "timing-models", cl::Optional,
    cl::desc(
        "Relative path to JSON-formatted file containing timing models for "
        "dataflow components. The tool will fetch the latency information from "
        "this file"),
    cl::init("data/components.json"), cl::cat(mainCategory));

static size_t findIndexInRange(ValueRange range, Value val) {
  for (auto [idx, res] : llvm::enumerate(range))
    if (res == val)
      return idx;
  llvm_unreachable("value should exist in range");
}

std::string getNodeLatencyAttr(Operation *op, TimingDatabase &timingDB) {
  double latency;
  if (failed(timingDB.getLatency(op, SignalType::DATA, latency)))
    return "0";
  return std::to_string(static_cast<unsigned>(latency));
}

LogicalResult getBufferImpl(bool transp, unsigned slots,
                            mlir::raw_indented_ostream &os) {

  if (!transp && (slots == 1 || slots == 2)) {
    os << "MODULE buffer" << slots << "o_1_1(dataIn0, pValid0, nReady0)\n";
    os << "VAR\n";
    os << "b0     : tehb_1_1(dataIn0, pValid0, b1.ready0);\n";
    os << "b1     : oehb_1_1(b0.dataOut0, b0.valid0, nReady0);\n";
    os << "DEFINE\n";
    os << "dataOut0 := b1.dataOut0;\n";
    os << "valid0 := b1.valid0;\n";
    os << "ready0 := b0.ready0;\n";
    return success();
  }

  if (transp && slots == 1) {
    os << "MODULE _buffer1t_1_1(dataIn0, pValid0, nReady0)\n";
    os << "VAR\n";
    os << "b0 : tehb_1_1(dataIn0, pValid0, nReady0);\n";
    os << "DEFINE\n";
    os << "dataOut0 := b0.dataOut0;\n";
    os << "valid0   := b0.valid0;\n";
    os << "ready0   := b0.ready0;\n";
    return success();
  }

  std::vector<std::string> data;
  std::vector<std::string> valid;
  std::vector<std::string> ready;

  // cascading tslots
  if (transp) {
    data.emplace_back("dataIn0");
    valid.emplace_back("pValid0");
    for (unsigned i = 0; i < slots; i++) {
      data.emplace_back("b" + itostr(i) + ".dataOut0");
      valid.emplace_back("b" + itostr(i) + ".valid0");
      ready.emplace_back("b" + itostr(i) + ".ready0");
    }
    ready.emplace_back("nReady");
    os << "MODULE buffer" << slots << "t_1_1(dataIn0, pValid0, nReady0)\n";
    os << "VAR\n";
    os << "DEFINE dataOut0 := b" << slots - 1 << ".dataOut0; \n";
    os << "DEFINE valid0   := b" << slots - 1 << ".valid0; \n";
    os << "DEFINE ready0   := b0.ready0; \n";
    for (unsigned i = 0; i < slots; i++) {
      os << "VAR b" << i << " : tslot_1_1(" << data[i] << ", " << valid[i]
         << ", " << ready[i] << ");\n";
    }
  }
  return success();
}

// for parametrized units, we encode the parameter in the type of the unit and
// later generate a unit for each parametrization used by the model
LogicalResult getNodeType(Operation *op, mlir::raw_indented_ostream &os,
                          TimingDatabase timingDB) {

  int numInputs = op->getNumOperands();
  int numOutputs = op->getNumResults();

  std::string type =
      llvm::TypeSwitch<Operation *, std::string>(op)
          .Case<handshake::ConstantOp>(
              [&](handshake::ConstantOp cstOp) -> std::string {
                return "constant";
              })
          .Case<handshake::OEHBOp>([&](handshake::OEHBOp oehbOp) {
            return "buffer" + std::to_string(oehbOp.getSlots()) + "o";
          })
          .Case<handshake::TEHBOp>([&](handshake::TEHBOp tehbOp) {
            return "buffer" + std::to_string(tehbOp.getSlots()) + "t";
          })
          .Case<handshake::ReturnOp>([&](auto) { return "buffer1t"; })
          .Case<arith::CmpIOp, arith::CmpFOp, arith::AndIOp, arith::OrIOp,
                arith::XOrIOp>([&](auto) {
            return "decider" + getNodeLatencyAttr(op, timingDB) + "c";
          })
          .Case<handshake::ForkOp>([&](auto) { return "fork"; })
          .Case<handshake::ControlMergeOp>([&](auto) { return "cmerge"; })
          .Case<handshake::SourceOp>([&](auto) { return "source"; })
          .Case<handshake::SinkOp>([&](auto) { return "sink"; })
          .Case<handshake::MuxOp>([&](auto) { return "mux"; })
          .Case<handshake::MergeOp>([&](auto) { return "merge"; })
          .Case<handshake::LSQOp>([&](auto) { return "lsq"; })
          .Case<handshake::MemoryControllerOp>([&](auto) { return "mc"; })
          .Case<handshake::ConditionalBranchOp, handshake::BranchOp>(
              [&](auto) { return "branch"; })
          .Case<arith::AddIOp, arith::AddFOp, arith::SubIOp, arith::SubFOp,
                arith::MulIOp, arith::MulFOp, arith::DivUIOp, arith::DivSIOp,
                arith::DivFOp, arith::ShRSIOp, arith::ShRUIOp, arith::ShLIOp,
                arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp, arith::TruncIOp,
                arith::TruncFOp>([&](auto) {
            return "operator" + getNodeLatencyAttr(op, timingDB) + "c";
          })
          .Default(
              [&](auto) { return "unknownop" + std::to_string(numInputs); });

  os << type << "_" << std::to_string(numInputs) << "_"
     << std::to_string(numOutputs);
  ;
  return success();
}

// prints the declaration for the unit
LogicalResult printUnit(Operation *op, mlir::raw_indented_ostream &os,
                        TimingDatabase &timingDB) {
  StringRef unitName = getUniqueName(op);

  os << "VAR " << unitName << " : ";
  if (failed(getNodeType(op, os, timingDB))) {
    return failure();
  }
  os << " (";

  for (auto [idx, arg] : llvm::enumerate(op->getOperands())) {
    os << unitName << "_dataIn" << idx << ", " << unitName << "_pValid" << idx;
    if (idx < op->getNumOperands() - 1) {
      os << ", ";
    }
  }

  if (op->getNumOperands() > 0 && op->getNumResults() > 0)
    os << ", ";

  for (auto [idx, arg] : llvm::enumerate(op->getResults())) {
    os << unitName << "_nReady" << idx;
    if (idx < op->getNumResults() - 1) {
      os << ", ";
    }
  }
  os << ");\n";
  return success();
}

// prints the declaration for the channels, one line per each data, ready and
// valid signal
LogicalResult printEdge(OpOperand &oprd, mlir::raw_indented_ostream &os) {
  Value val = oprd.get();
  Operation *src = val.getDefiningOp();
  Operation *dst = oprd.getOwner();

  // Locate value in source results and destination operands
  const size_t resIdx = findIndexInRange(src->getResults(), val);
  const size_t argIdx = findIndexInRange(dst->getOperands(), val);

  std::string cstValue =
      llvm::TypeSwitch<Operation *, std::string>(src)
          .Case<handshake::ConstantOp>([&](handshake::ConstantOp cstOp) {
            Type cstType = cstOp.getResult().getType();
            TypedAttr valueAttr = cstOp.getValueAttr();
            if (isa<IntegerType>(cstType)) {
              if (auto boolAttr = dyn_cast<mlir::BoolAttr>(valueAttr))
                return boolAttr.getValue() ? "TRUE" : "FALSE";
            }
            // if not a Boolean value: the abstract model doesn't care constant
            // values like this
            return "FALSE";
          })
          .Default([&](auto) { return "false"; });

  os << "DEFINE " << getUniqueName(src) << "_nReady" << resIdx << " = "
     << getUniqueName(dst) << ".ready" << argIdx << ";\n";

  os << "DEFINE " << getUniqueName(dst) << "_pValid" << argIdx << " = "
     << getUniqueName(src) << ".valid" << resIdx << ";\n";

  if (!isa<handshake::ConstantOp>(src))
    os << "DEFINE " << getUniqueName(dst) << "_dataIn" << argIdx << " = "
       << getUniqueName(src) << ".dataOut" << resIdx << ";\n";
  else
    os << "DEFINE " << getUniqueName(dst) << "_dataIn" << argIdx << " = "
       << cstValue << ";\n";

  return success();
}

LogicalResult writeSmv(mlir::ModuleOp mod, TimingDatabase &timingDB) {
  mlir::raw_indented_ostream stdOs(llvm::outs());
  auto funcs = mod.getOps<handshake::FuncOp>();
  if (++funcs.begin() != funcs.end()) {
    mod->emitOpError()
        << "we currently only support one handshake function per module";
    return failure();
  }
  NameAnalysis nameAnalysis = NameAnalysis(mod);
  if (!nameAnalysis.isAnalysisValid())
    return failure();
  nameAnalysis.nameAllUnnamedOps();

  handshake::FuncOp funcOp = *funcs.begin();

  stdOs << "MODULE main\n";
  stdOs << "\n-- units\n";

  for (auto &op : funcOp.getOps()) {
    if (failed(printUnit(&op, stdOs, timingDB)))
      return failure();
  }

  stdOs << "\n\n-- channels\n";

  for (auto &op : funcOp.getOps()) {
    for (OpResult res : op.getResults()) {
      for (OpOperand &val : res.getUses()) {
        if (failed(printEdge(val, stdOs)))
          return failure();
      }
    }
  }

  stdOs << "\n\n-- formal properties\n";

  return success();
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(
      argc, argv,
      "Exports a SMV model corresponding to the module"
      "The tool only supports exporting the graph of a single Handshake "
      "function at the moment, and will fail"
      "if there is more than one Handhsake function in the module.");

  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(inputFileName.c_str());
  if (std::error_code error = fileOrErr.getError()) {
    llvm::errs() << argv[0] << ": could not open input file '" << inputFileName
                 << "': " << error.message() << "\n";
    return 1;
  }

  // Functions feeding into HLS tools might have attributes from high(er)
  // level dialects or parsers. Allow unregistered dialects to not fail in
  // these cases
  MLIRContext context;
  context.loadDialect<memref::MemRefDialect, arith::ArithDialect,
                      handshake::HandshakeDialect, math::MathDialect>();
  context.allowUnregisteredDialects();

  TimingDatabase timingDB(&context);
  if (failed(TimingDatabase::readFromJSON(timingDBFilepath, timingDB))) {
    llvm::errs() << "Failed to read timing database at \"" << timingDBFilepath
                 << "\"\n";
    return 1;
  }

  // Load the MLIR module
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> mod(
      mlir::parseSourceFile<ModuleOp>(sourceMgr, &context));
  if (!mod)
    return 1;
  return failed(writeSmv(*mod, timingDB));
}
