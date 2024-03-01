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
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include <cstddef>
#include <set>
#include <string>
#include <tuple>
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

static StringRef getMemName(Value memref) {
  Operation *parentOp = memref.getParentBlock()->getParentOp();
  handshake::FuncOp funcOp = dyn_cast<handshake::FuncOp>(parentOp);
  for (auto [name, funArg] :
       llvm::zip(funcOp.getArgNames(), funcOp.getArguments())) {
    if (funArg == memref)
      return cast<StringAttr>(name).getValue();
  }
  return StringRef();
}

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

  os << "\n";
  os << "MODULE buffer" << slots << (transp ? "t" : "o")
     << "_1_1(dataIn0, pValid0, nReady0)\n";
  os << "VAR\n";

  // Currently, when mapping the (transp, slot) to the buffer implementation, we
  // consider the following convention:
  // not transparent && slot == 1 or 2: OEHB
  //     transparent && slot == 1     : TEHB
  // not transparent && slot >  2     : OEHB + tslots * (slot - 1) + TEHB
  //     transparent && slot >  1     : tslots * (slot)

  // This will be revised in the future when we consider the buffer parameter as
  // a tuple, i.e., (fwdLatency, bwdLatency, slots)

  if (!transp && (slots == 1 || slots == 2)) {
    os << "b0     : tehb_1_1(dataIn0, pValid0, b1.ready0);\n";
    os << "b1     : oehb_1_1(b0.dataOut0, b0.valid0, nReady0);\n";
    os << "DEFINE\n";
    os << "dataOut0 := b1.dataOut0;\n";
    os << "valid0 := b1.valid0;\n";
    os << "ready0 := b0.ready0;\n";
    return success();
  }

  if (transp && slots == 1) {
    os << "b0 : tehb_1_1(dataIn0, pValid0, nReady0);\n";
    os << "DEFINE\n";
    os << "dataOut0 := b0.dataOut0;\n";
    os << "valid0   := b0.valid0;\n";
    os << "ready0   := b0.ready0;\n";
    return success();
  }

  // The data/valid/ready port signals of the individual slots
  std::vector<std::string> data;
  std::vector<std::string> valid;
  std::vector<std::string> ready;

  // Transparent buffer with more than 1 slot: equivalent to a N-slot elastic
  // FIFO with bypass, i.e, has no sequential delay on both directions.
  data.emplace_back("dataIn0");
  valid.emplace_back("pValid0");
  for (unsigned i = 0; i < slots - 1 + (!transp); i++) {
    data.emplace_back("b" + itostr(i) + ".dataOut0");
    valid.emplace_back("b" + itostr(i) + ".valid0");
    ready.emplace_back("b" + itostr(i) + ".ready0");
  }
  ready.emplace_back("nReady");
  os << "DEFINE dataOut0 := b" << slots - (transp) << ".dataOut0; \n";
  os << "DEFINE valid0   := b" << slots - (transp) << ".valid0; \n";
  os << "DEFINE ready0   := b0.ready0; \n";
  if (transp) {
    assert(std::size(data) == slots);
    for (unsigned i = 0; i < slots; i++) {
      os << "VAR b" << i << " : tslot_1_1(" << data[i] << ", " << valid[i]
         << ", " << ready[i] << ");\n";
    }
    return success();
  }
  // Non-transparent buffer (N slots) with more than 2 slots: equivalent to a
  // N-slot elastic FIFO + TEHB, i.e., it has sequential latency on both
  // directions
  if (!transp) {
    assert(std::size(data) == slots + 1);
    os << "VAR b0 : oehb_1_1(" << data[0] << ", " << valid[0] << ", "
       << ready[0] << ");\n";
    for (unsigned i = 1; i < slots; i++) {
      os << "VAR b" << i << " : tslot_1_1(" << data[i] << ", " << valid[i]
         << ", " << ready[i] << ");\n";
    }
    os << "VAR b" << slots << " : tehb_1_1" << data[slots] << ", "
       << valid[slots] << ", " << ready[slots] << ");\n";
    return success();
  }

  // encountered unknown buffer configuration!
  return failure();
}

LogicalResult getForkImpl(unsigned nOutputs, mlir::raw_indented_ostream &os) {
  assert(nOutputs >= 2 && "received a fork with 0 or 1 output ports!");
  // interface
  os << "\nMODULE fork_1_" << nOutputs << "(dataIn0, pValid0, ";
  for (unsigned i = 0; i < nOutputs - 1; i++) {
    os << "nReady" << i << ", ";
  }
  os << "nReady" << nOutputs - 1 << ")\n";
  os << "forkStop := ";
  for (unsigned i = 0; i < nOutputs - 1; i++) {
    os << "regBlock" << i << ".blockStop | ";
  }
  os << "regBlock" << nOutputs - 1 << "blockStop;\n";
  os << "pValidAndForkStop := pValid0 & forkStop;\n";
  os << "ready0 := !forkStop;\n";
  for (unsigned i = 0; i < nOutputs; i++) {
    os << "-- output" << i << "\n";
    os << "VAR regBlock" << i << " : eagerFork_RegisterBlock(pValid0, !nReady"
       << i << ", pValidAndForkStop);\n";
    os << "DEFINE valid" << i << " := regBlock" << i << ".valid;\n";
    os << "DEFINE dataOut" << i << " := dataIn0;\n";
    os << "sent" << i << " := !regBlock" << i << ".reg_value;\n";
    os << "sent_plus" << i << " := !regBlock" << i << ".reg_value & dataIn0;\n";
    os << "sent_minus" << i << " := !regBlock" << i
       << ".reg_value & !dataIn0;\n";
  }

  return success();
}

LogicalResult getOperatorImpl(unsigned numInputs, unsigned lat,
                              raw_indented_ostream &os) {
  os << "\nMODULE operator" << lat << "c_" << numInputs << "_1 (";
  for (unsigned i = 0; i < numInputs; i++) {
    os << "dataIn" << i << ", pValid" << i << ", ";
  }
  os << "nReady0)\nVAR\n";
  os << "d0 : delay" << lat << "c_1_1(FALSE, j0.valid0, nReady0);\n";
  os << "j0 : join_" << numInputs << "_1(";
  for (unsigned i = 0; i < numInputs; i++) {
    os << "pValid" << i << ", ";
  }
  os << "d0.ready0);\n";
  os << "DEFINE\n";
  os << "valid0   := d0.valid0;\n";
  os << "dataOut0 := d0.dataOut0;\n";
  os << "num      := d0.num;\n";
  for (unsigned i = 0; i < numInputs; i++) {
    os << "ready" << i << " := j0.ready" << i << ";\n";
  }
  return success();
}

// this implements model of the pipeline stage inside the pipelined units
LogicalResult getDelayImpl(unsigned lat, raw_indented_ostream &os) {
  os << "\nMODULE delay" << lat << "c_1_1 (dataIn0, pValid0, nReady0)\n";
  if (lat == 0) {
    os << "DEFINE dataOut0 := dataIn0;\n";
    os << "DEFINE valid0   := pValid0;\n";
    os << "DEFINE ready0   := nReady0;\n";
    return success();
  }

  if (lat == 1) {
    os << "VAR b0          := oehb_1_1(dataIn0, pValid0, nReady0);\n";
    os << "DEFINE dataOut0 := b0.dataIn0;\n";
    os << "DEFINE valid0   := b0.pValid0;\n";
    os << "DEFINE ready0   := b0.ready0;\n";
    os << "DEFINE num      := toint(b0.valid0);\n";
    os << "DEFINE v1.full  := b0.valid0;\n";
    os << "DEFINE v1.num   := num;\n";
    return success();
  }

  if (lat > 1) {
    os << "VAR b0 := oehb_1_1(dataIn0, v" << lat - 1 << ", nReady0);\n";
    os << "DEFINE v0       := pValid0;\n";
    for (unsigned i = 1; i < lat; i++) {
      os << "VAR b" << i << " : boolean;\n";
      os << "ASSIGN init(v" << i << ") := FALSE;\n";
      os << "ASSIGN next(v" << i << ") := b0.ready0 ? v" << i - 1 << " : v" << i
         << ";\n";
    }
    os << "DEFINE dataOut0 := FALSE;\n";
    os << "DEFINE valid0   := b0.pValid0;\n";
    os << "DEFINE ready0   := b0.ready0;\n";
    os << "DEFINE v" << lat << " := b0.valid0;\n";

    os << "num : count(";
    for (unsigned i = 1; i < lat; i++) {
      os << "v" << itostr(i) << ", ";
    }
    os << "v" << lat << ");\n";
    for (unsigned i = 1; i < lat; i++) {
      os << "DEFINE v" << i << ".full := v" << i << ";\n";
      os << "DEFINE v" << i << ".num  := toint(v" << i << ");\n";
    }
    return success();
  }

  return failure();
}

// for parametrized units, we encode the parameter in the type of the unit and
// later generate a unit for each parametrization used by the model
LogicalResult getUnitType(Operation *op, mlir::raw_indented_ostream &os,
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
  if (failed(getUnitType(op, os, timingDB))) {
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
LogicalResult printChannel(OpOperand &oprd, mlir::raw_indented_ostream &os) {
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

  os << "DEFINE " << getUniqueName(src) << "_nReady" << resIdx
     << " := " << getUniqueName(dst) << ".ready" << argIdx << ";\n";

  os << "DEFINE " << getUniqueName(dst) << "_pValid" << argIdx
     << " := " << getUniqueName(src) << ".valid" << resIdx << ";\n";

  if (!isa<handshake::ConstantOp>(src))
    os << "DEFINE " << getUniqueName(dst) << "_dataIn" << argIdx
       << " := " << getUniqueName(src) << ".dataOut" << resIdx << ";\n";
  else
    os << "DEFINE " << getUniqueName(dst) << "_dataIn" << argIdx
       << " := " << cstValue << ";\n";

  return success();
}

LogicalResult writeUnitImpl(handshake::FuncOp &funcOp,
                            mlir::raw_indented_ostream &os,
                            const TimingDatabase &timingDB) {
  // make sure the each configuration is generated no more than once
  std::set<std::tuple<bool, signed>> bufAttrs;
  std::set<unsigned> forkAttrs, delayAttrs;
  std::set<std::tuple<unsigned, unsigned>> operatorAttrs;
  for (auto &op : funcOp.getOps()) {
    if (handshake::OEHBOp oehb = llvm::dyn_cast<handshake::OEHBOp>(op); oehb) {
      unsigned slots = oehb.getSlots();
      if (bufAttrs.count({false, slots}) == 0) {
        if (failed(getBufferImpl(false, slots, os)))
          return failure();
      }
      bufAttrs.emplace(false, slots);
    }
    if (handshake::TEHBOp tehb = llvm::dyn_cast<handshake::TEHBOp>(op); tehb) {
      unsigned slots = tehb.getSlots();
      if (bufAttrs.count({true, slots}) == 0) {
        if (failed(getBufferImpl(true, slots, os)))
          return failure();
        bufAttrs.emplace(true, slots);
      }
    }
    if (handshake::ForkOp forkOp = llvm::dyn_cast<handshake::ForkOp>(op);
        forkOp) {
      unsigned numOutputs = forkOp.getNumResults();
      // only generate implementation for forks that have more than 3 outputs
      // the 2-output and 3-output forks are provided in the smv component
      // library
      if (forkAttrs.count(numOutputs) == 0 && numOutputs > 3) {
        if (failed(getForkImpl(numOutputs, os)))
          return failure();
        forkAttrs.emplace(numOutputs);
      }
    }
    if (isa<arith::AddIOp, arith::AddFOp, arith::SubIOp, arith::SubFOp,
            arith::MulIOp, arith::MulFOp, arith::DivUIOp, arith::DivSIOp,
            arith::DivFOp, arith::ShRSIOp, arith::ShRUIOp, arith::ShLIOp,
            arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp, arith::TruncIOp,
            arith::TruncFOp>(op)) {
      double floatLat = 0.0;
      unsigned lat = 0;
      if (!failed(timingDB.getLatency(&op, SignalType::DATA, floatLat))) {
        lat = (int)round(floatLat);
      }
      if (delayAttrs.count(lat) == 0) {
        if (failed(getDelayImpl(lat, os)))
          return failure();
        delayAttrs.emplace((int)round(lat));
      }
      int numInputs = op.getNumOperands();
      if (operatorAttrs.count({numInputs, lat}) == 0) {
        if (failed(getOperatorImpl(numInputs, lat, os)))
          return failure();
        operatorAttrs.emplace(numInputs, lat);
      }
    }
  }
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
        if (failed(printChannel(val, stdOs)))
          return failure();
      }
    }
  }

  stdOs << "\n\n-- formal properties\n";

  stdOs << "\n\n-- parametrized units\n";
  if (failed(writeUnitImpl(funcOp, stdOs, timingDB)))
    return failure();

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
