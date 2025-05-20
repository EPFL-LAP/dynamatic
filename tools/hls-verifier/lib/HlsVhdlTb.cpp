//===- HlsVhdlTb.cpp --------------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <optional>
#include <string>

#include "HlsVhdlTb.h"
#include "VerificationContext.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"

using std::tuple;

// A Helper struct for connecting dynamatic's MemRef argument to a two-port RAM.
// This grouping makes codegen more consistent in their code style, and
// centralize the specialize handling in this struct.
// This class provides:
// - declareConstants: declare the generics for the dual port RAM
// - declareSignals: signals between the dual port RAM and the circuit
// - instantiateRAMModel: instantiate the RAMs that communicate with files
// - connectToDuv: connect the RAM to the circuit
//
// Current assumption: the circuit is always a master device w.r.t. the RAM
//
// This struct can be extended / adapted to other interfaces like AXI
struct MemRefToDualPortRAM {
  SmallVector<tuple</* Signal name of the circuit */ std::string,
                    /* Signal name of the DPRAM*/ std::string,
                    /* Bitwidth */ unsigned>>
      memrefToDPRAM;
  mlir::MemRefType type;
  std::string argName;

  MemRefToDualPortRAM(mlir::MemRefType type, const std::string &argName)
      : type(type), argName(argName) {

    int dataWidth = type.getElementType().getIntOrFloatBitWidth();
    int dataDepth = type.getNumElements();
    int addrWidth = ((int)ceil(log2(dataDepth)));

    // The two_port_RAM has two read/write interfaces, each has
    // - we: write enable (must be set to 1 when writing)
    // - ce: clock enable (must be set to 1 for both read and write).
    // - address: address port
    // - din: store data
    // - dout: read data
    // Currently, our memory controler assumes that there is only one read
    // channel and one write channel. Therefore, many signals are not used

    // This mapping only records the used signals.
    memrefToDPRAM.emplace_back("storeEn", WE0_PORT, 1);
    memrefToDPRAM.emplace_back("storeData", D_IN0_PORT, dataWidth);
    memrefToDPRAM.emplace_back("storeAddr", ADDR0_PORT, addrWidth);
    memrefToDPRAM.emplace_back("loadEn", CE1_PORT, 1);
    memrefToDPRAM.emplace_back("loadData", D_OUT1_PORT, dataWidth);
    memrefToDPRAM.emplace_back("loadAddr", ADDR1_PORT, addrWidth);
  }

  void declareConstants(mlir::raw_indented_ostream &os,
                        const std::string &inputVectorPath,
                        const std::string &outputFilePath) {
    int dataWidth = type.getElementTypeBitWidth();
    int dataDepth = type.getNumElements();
    int addrWidth = ((int)ceil(log2(dataDepth)));
    declareConstant(os, "INPUT_" + argName, "STRING",
                    "\"" + inputVectorPath + "/input_" + argName + ".dat" +
                        "\"");
    declareConstant(os, "OUTPUT_" + argName, "STRING",
                    "\"" + outputFilePath + "/output_" + argName + ".dat" +
                        "\"");
    declareConstant(os, "DATA_WIDTH_" + argName, "INTEGER",
                    to_string(dataWidth));
    declareConstant(os, "ADDR_WIDTH_" + argName, "INTEGER",
                    to_string(addrWidth));
    declareConstant(os, "DATA_DEPTH_" + argName, "INTEGER",
                    to_string(dataDepth));
  }

  // Declare signals appear in the circuit interface
  void declareSignals(mlir::raw_indented_ostream &os) {
    for (auto &[_, sigName, bitwidth] : memrefToDPRAM) {
      if (sigName == WE0_PORT or sigName == CE1_PORT) {
        declareSTL(os, argName + "_" + sigName, std::nullopt);
      } else {
        declareSTL(os, argName + "_" + sigName, to_string(bitwidth));
      }
    }
    int dataWidth = type.getElementTypeBitWidth();
    // Declare unused interfaces in the two port RAM
    declareSTL(os, argName + "_" + D_OUT0_PORT, to_string(dataWidth));
    declareSTL(os, argName + "_" + D_IN1_PORT, to_string(dataWidth),
               "(others => \'0\')");
    // The write enable of the read interface is not used
    declareSTL(os, argName + "_" + WE1_PORT, nullopt, "\'0\'");
    // The read enable of the write interface is not used
    declareSTL(os, argName + "_" + CE0_PORT, nullopt, "\'1\'");
  }

  void instantiateRAMModel(mlir::raw_indented_ostream &os) {
    Instance memInst("two_port_RAM", "mem_inst_" + argName);

    memInst.parameter(IN_FILE_PARAM, "INPUT_" + argName)
        .parameter(OUT_FILE_PARAM, "OUTPUT_" + argName)
        .parameter(DATA_WIDTH_PARAM, "DATA_WIDTH_" + argName)
        .parameter(ADDR_WIDTH_PARAM, "ADDR_WIDTH_" + argName)
        .parameter(DATA_DEPTH_PARAM, "DATA_DEPTH_" + argName)
        .connect(CLK_PORT, "tb_" + CLK_PORT)
        .connect(RST_PORT, "tb_" + RST_PORT)
        .connect(DONE_PORT, "tb_stop");
    for (auto &[_, portName, bitwidth] : memrefToDPRAM) {
      memInst.connect(portName, argName + "_" + portName);
    }

    // Declare unused interfaces in the two port RAM
    memInst.connect(D_OUT0_PORT, argName + "_" + D_OUT0_PORT);
    memInst.connect(D_IN1_PORT, argName + "_" + D_IN1_PORT);
    memInst.connect(WE1_PORT, argName + "_" + WE1_PORT);
    memInst.connect(CE0_PORT, "\'1\'");

    memInst.emitVhdl(os);
  }

  void connectToDuv(Instance &duvInst) {
    for (auto &[portSuffix, sigSuffix, bitwidth] : memrefToDPRAM)
      duvInst.connect(argName + "_" + portSuffix, argName + "_" + sigSuffix);
  }
};

// A helper struct for connecting the in/output channels from/to the single
// argument.  The single argument block is basically an one-element RAM, with
// the following interface signals:
// - ce0: clock enable
// - we0: if doing the write access
// - din0: data input
// - dout0: data output
// - dout0_valid: output handshake
// - dout0_ready: output handshake
struct ChannelToSingleArgument {

  handshake::ChannelType type;
  std::string argName;
  bool isInput;

  ChannelToSingleArgument(handshake::ChannelType type,
                          const std::string &argName, bool isInput)
      : type(type), argName(argName), isInput(isInput) {}

  void declareConstants(mlir::raw_indented_ostream &os,
                        const std::string &inputVectorPath,
                        const std::string &outputFilePath) {

    // If the single argument is an output (e.g., the return value), we don't
    // specify any input vector file ("").
    std::string inputFile =
        isInput ? "\"" + inputVectorPath + "/input_" + argName + ".dat" + "\""
                : "\"\"";
    declareConstant(os, "INPUT_" + argName, "STRING", inputFile);

    declareConstant(os, "OUTPUT_" + argName, "STRING",
                    "\"" + outputFilePath + "/output_" + argName + ".dat" +
                        "\"");
    int dataWidth = type.getDataBitWidth();
    declareConstant(os, "DATA_WIDTH_" + argName, "INTEGER",
                    to_string(dataWidth));
  }

  void declareSignals(mlir::raw_indented_ostream &os) {
    unsigned dataWidth = type.getDataBitWidth();
    if (not isInput) {
      // The valid signal from the circuit, which drives the write enable (we)
      // pin of the single enable module
      declareSTL(os, argName + "_valid");
      declareSTL(os, argName + "_ready");
    }
    // The single argument block needs to define all these signals, regardless
    // of using as an input argument or an output argument
    declareSTL(os, argName + "_" + CE0_PORT);
    declareSTL(os, argName + "_" + WE0_PORT);

    declareSTL(os, argName + "_din0", to_string(dataWidth));

    declareSTL(os, argName + "_dout0", to_string(dataWidth));
    declareSTL(os, argName + "_dout0_valid");
    declareSTL(os, argName + "_dout0_ready");
  }

  void instantiateSingleArgumentModel(mlir::raw_indented_ostream &os) {
    Instance argInst("single_argument", "arg_inst_" + argName);

    argInst.parameter(IN_FILE_PARAM, "INPUT_" + argName)
        .parameter(OUT_FILE_PARAM, "OUTPUT_" + argName)
        .parameter(DATA_WIDTH_PARAM, "DATA_WIDTH_" + argName)
        // NOTE: The lhs of the connect (e.g., CLK_PORT) is the port name of
        // the instantiated RAM models (e.g., two_port_RAM.vhd and
        // single_argument.vhd). This name is not necessarily named in the
        // same way as the testbench signal (e.g., tb_clk). For Dynamatic's
        // internal coverification, we have the freedom to name the
        // interface of the RAM model however we want.
        .connect(CLK_PORT, "tb_" + CLK_PORT)
        .connect(RST_PORT, "tb_" + RST_PORT)
        .connect(DONE_PORT, "tb_temp_idle")
        .connect(D_OUT0_PORT, argName + "_dout0")
        .connect(D_OUT0_PORT + "_valid", argName + "_dout0_valid")
        .connect(D_OUT0_PORT + "_ready", argName + "_dout0_ready");
    if (isInput) {
      argInst.connect(CE0_PORT, "'1'")
          .connect(WE0_PORT, "'0'")
          .connect(D_IN0_PORT, "(others => '0')");
    } else {
      argInst.connect(CE0_PORT, "'1'")
          .connect(WE0_PORT, argName + "_valid")
          .connect(D_IN0_PORT, argName + "_din0");
    }
    argInst.emitVhdl(os);
  }

  void connectToDuv(Instance &duvInst) {
    if (isInput) {
      duvInst.connect(argName, argName + "_dout0")
          .connect(argName + "_valid", argName + "_dout0_valid")
          .connect(argName + "_ready", argName + "_dout0_ready");
    } else {
      duvInst.connect(argName, argName + "_din0")
          .connect(argName + "_valid", argName + "_valid")
          .connect(argName + "_ready", argName + "_ready");
    }
  }
};

// A helper construct for interfacing the control only signals of the DUV
//
// HACK: the control only signals are handled in the following ways:
// - Input: they are driven by valid = constant 1, their ready signals are
// ignored
// - Output: they are driving the join
struct ControlOnlyConnector {

  handshake::ControlType type;
  std::string argName;
  bool isInput;

  ControlOnlyConnector(handshake::ControlType type, const std::string &argName,
                       bool isInput)
      : type(type), argName(argName), isInput(isInput) {}

  void declareSignals(mlir::raw_indented_ostream &os) {
    declareSTL(os, argName + "_valid");
    declareSTL(os, argName + "_ready");
  }

  void connectToDuv(Instance &duvInst) {
    if (isInput) {
      duvInst.connect(argName + "_valid", "\'1\'")
          .connect(argName + "_ready", argName + "_ready");
    } else {
      duvInst.connect(argName + "_valid", argName + "_valid")
          .connect(argName + "_ready", argName + "_ready");
    }
  }
};

// function to get the port name in the entity for each paramter
void getConstantDeclaration(mlir::raw_indented_ostream &os,
                            VerificationContext &ctx) {

  handshake::FuncOp *funcOp = ctx.funcOp;

  std::string inputVectorPath = ctx.getInputVectorDir();
  std::string outputFilePath = ctx.getHdlOutDir();

  // The files and configuration of the single_argument model of the data input
  // channels
  for (auto &[type, argName] :
       getInputArguments<handshake::ChannelType>(funcOp)) {

    ChannelToSingleArgument c(type, argName, /* isInput = */ true);
    c.declareConstants(os, inputVectorPath, outputFilePath);
  }

  // The files and configuration of the two port RAM model of the arrays
  for (auto &[type, argName] : getInputArguments<mlir::MemRefType>(funcOp)) {
    MemRefToDualPortRAM m(type, argName);
    m.declareConstants(os, inputVectorPath, outputFilePath);
  }

  // The files and configuration of the single_argument model of the data output
  // channels
  for (auto &[type, argName] :
       getOutputArguments<handshake::ChannelType>(funcOp)) {
    ChannelToSingleArgument c(type, argName, /* isInput = */ false);
    c.declareConstants(os, inputVectorPath, outputFilePath);
  }
  declareConstant(os, "HALF_CLK_PERIOD", "TIME", "2.00 ns");
  declareConstant(os, "TRANSACTION_NUM", "INTEGER", to_string(1));
}

// This writes the signal declarations fot the testbench
// Example:
// signal tb_clk : std_logic := '0';
// signal tb_start_valid : std_logic := '0';
void getSignalDeclaration(mlir::raw_indented_ostream &os,
                          VerificationContext &ctx) {

  handshake::FuncOp *funcOp = ctx.funcOp;

  declareSTL(os, "tb_clk", std::nullopt, "'0'");
  declareSTL(os, "tb_start_valid", std::nullopt, "'0'");
  declareSTL(os, "tb_rst", std::nullopt, "'0'");
  declareSTL(os, "tb_started");
  declareSTL(os, "tb_global_valid");
  declareSTL(os, "tb_global_ready");
  declareSTL(os, "tb_stop");

  // Signals of data input channels
  for (auto &[type, argName] :
       getInputArguments<handshake::ChannelType>(funcOp)) {
    ChannelToSingleArgument c(type, argName, /* isInput = */ true);
    c.declareSignals(os);
  }

  // Signals of control input channels
  for (auto &[type, argName] :
       getInputArguments<handshake::ControlType>(funcOp)) {
    ControlOnlyConnector c(type, argName, /* isInput = */ true);
    c.declareSignals(os);
  }

  // Signals of the memory reference signals
  for (auto &[type, argName] : getInputArguments<mlir::MemRefType>(funcOp)) {

    MemRefToDualPortRAM m(type, argName);
    m.declareSignals(os);
  }

  // Signals of data output channels
  for (auto &[type, argName] :
       getOutputArguments<handshake::ChannelType>(funcOp)) {
    ChannelToSingleArgument c(type, argName, /* isInput = */ false);
    c.declareSignals(os);
  }

  // Signals of control output channels
  for (auto &[type, argName] :
       getOutputArguments<handshake::ControlType>(funcOp)) {
    ControlOnlyConnector c(type, argName, /* isInput = */ false);
    c.declareSignals(os);
  }

  os << "\n";

  declareSTL(os, "tb_temp_idle", std::nullopt, "'1'");
  os << "shared variable transaction_idx : INTEGER := 0;\n";
  os.flush();
}

void getMemoryInstanceGeneration(mlir::raw_indented_ostream &os,
                                 VerificationContext &ctx) {

  handshake::FuncOp *funcOp = ctx.funcOp;

  // Instantiate argument generator for the input channels
  for (auto &[type, argName] :
       getInputArguments<handshake::ChannelType>(funcOp)) {
    ChannelToSingleArgument c(type, argName, /* isInput = */ true);
    c.instantiateSingleArgumentModel(os);
  }

  // Instantiate dual port RAMs for the memory interfaces
  for (auto &[type, argName] : getInputArguments<mlir::MemRefType>(funcOp)) {
    MemRefToDualPortRAM m(type, argName);
    m.instantiateRAMModel(os);
  }

  for (auto &[type, argName] :
       getOutputArguments<handshake::ChannelType>(funcOp)) {
    ChannelToSingleArgument c(type, argName, /* isInput = */ false);
    c.instantiateSingleArgumentModel(os);
  }
}

void getDuvInstanceGeneration(mlir::raw_indented_ostream &os,
                              VerificationContext &ctx) {

  std::string duvName = ctx.vhdlDUVEntityName;

  Instance duvInst(duvName, "duv_inst");

  duvInst.connect(CLK_PORT, "tb_" + CLK_PORT)
      .connect(RST_PORT, "tb_" + RST_PORT);

  handshake::FuncOp *funcOp = ctx.funcOp;

  // Connect the input data channels to the DUV
  for (auto &[type, argName] :
       getInputArguments<handshake::ChannelType>(funcOp)) {
    ChannelToSingleArgument c(type, argName, /* isInput = */ true);
    c.connectToDuv(duvInst);
  }

  // Connect the input control channels to the DUV
  // @Jiahui17: TODO: create a new argument for dataless input channels
  for (auto &[type, argName] :
       getInputArguments<handshake::ControlType>(funcOp)) {
    ControlOnlyConnector c(type, argName, /* isInput = */ true);
    c.connectToDuv(duvInst);
  }

  // Connect the memory elements to the DUV
  for (auto &[type, argName] : getInputArguments<mlir::MemRefType>(funcOp)) {
    MemRefToDualPortRAM m(type, argName);
    m.connectToDuv(duvInst);
  }

  // Connect the input control channels to the DUV
  // @Jiahui17: TODO: create a new argument for dataless input channels
  for (auto &[type, argName] :
       getOutputArguments<handshake::ChannelType>(funcOp)) {
    ChannelToSingleArgument c(type, argName, /* isInput = */ false);
    c.connectToDuv(duvInst);
  }

  for (auto &[type, argName] :
       getOutputArguments<handshake::ControlType>(funcOp)) {
    ControlOnlyConnector c(type, argName, /* isInput = */ false);
    c.connectToDuv(duvInst);
  }

  duvInst.emitVhdl(os);
}

void deriveGlobalCompletionSignal(mlir::raw_indented_ostream &os,
                                  VerificationContext &ctx) {
  handshake::FuncOp *funcOp = ctx.funcOp;
  // @Jiahui17: I assume that the results only contain handshake channels.
  unsigned joinSize = funcOp->getNumResults();

  unsigned idx = 0;

  Instance joinInst("tb_join", "join_valids");

  joinInst.parameter("SIZE", std::to_string(joinSize));

  for (auto [resType, portAttr] :
       llvm::zip_equal(funcOp->getResultTypes(), funcOp->getResNames())) {
    std::string argName = portAttr.dyn_cast<StringAttr>().str();
    if (isa<handshake::ChannelType, handshake::ControlType>(resType)) {
      joinInst.connect("ins_valid(" + std::to_string(idx) + ")",
                       argName + "_valid");
      joinInst.connect("ins_ready(" + std::to_string(idx++) + ")",
                       argName + "_ready");
    }
  }
  joinInst.connect("outs_valid", "tb_global_valid");
  joinInst.connect("outs_ready", "tb_global_ready");

  joinInst.emitVhdl(os);
}

void getOutputTagGeneration(mlir::raw_indented_ostream &os,
                            VerificationContext &ctx) {
  handshake::FuncOp *funcOp = ctx.funcOp;

  // Reading / Dumping the content of the memory into the file
  for (auto &[type, argName] : getInputArguments<mlir::MemRefType>(funcOp)) {
    os << llvm::formatv(PROC_WRITE_TRANSACTIONS.c_str(), argName);
  }

  // Reading / Dumping the content of the memory into the file
  for (auto &[type, argName] :
       getInputArguments<handshake::ChannelType>(funcOp)) {
    os << llvm::formatv(PROC_WRITE_TRANSACTIONS.c_str(), argName);
  }

  // Reading / Dumping the content of the memory into the file
  for (auto &[type, argName] :
       getOutputArguments<handshake::ChannelType>(funcOp)) {
    os << llvm::formatv(PROC_WRITE_TRANSACTIONS.c_str(), argName);
  }
}

void vhdlTbCodegen(VerificationContext &ctx) {

  std::error_code ec;
  llvm::raw_fd_ostream fileStream(ctx.getVhdlTestbenchPath(), ec);
  if (ec) {
    llvm::errs() << "Error opening file: " << ec.message() << "\n";
    // Handle error appropriately, e.g., return, exit, etc.
    assert(false);
  }
  mlir::raw_indented_ostream os(fileStream);

  os << VHDL_LIBRARY_HEADER;
  os << "entity tb is\n";
  os << "end entity tb;\n\n";
  os << "architecture behavior of tb is\n\n";
  os.indent();
  getConstantDeclaration(os, ctx);
  getSignalDeclaration(os, ctx);
  os.unindent();
  os << "begin\n\n";
  os.indent();
  getDuvInstanceGeneration(os, ctx);
  getMemoryInstanceGeneration(os, ctx);
  deriveGlobalCompletionSignal(os, ctx);
  getOutputTagGeneration(os, ctx);
  os << COMMON_TB_BODY;
  os.unindent();
  os << "end architecture behavior;\n";
  os.flush();
}
