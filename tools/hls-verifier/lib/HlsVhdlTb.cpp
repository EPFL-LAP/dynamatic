//===- HlsVhdlTb.cpp --------------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>

#include "CAnalyser.h"
#include "HlsLogging.h"
#include "HlsVhdlTb.h"
#include "VerificationContext.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"

namespace hls_verify {
const string LOG_TAG = "VVER";

static const string VHDL_LIBRARY_HEADER = R"DELIM(
library IEEE;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_textio.all;
use ieee.numeric_std.all;
use std.textio.all;
use work.sim_package.all;
)DELIM";

static const string COMMON_TB_BODY = R"DELIM(

generate_sim_done_proc : process
begin
  while (transaction_idx /= TRANSACTION_NUM) loop
    wait until tb_clk'event and tb_clk = '1';
  end loop;
  wait until tb_clk'event and tb_clk = '1';
  wait until tb_clk'event and tb_clk = '1';
  wait until tb_clk'event and tb_clk = '1';
  assert false
  report "simulation done!"
  severity note;
  assert false
  report "NORMAL EXIT (note: failure is to force the simulator to stop)"
  severity failure;
  wait;
end process;

gen_clock_proc : process
begin
  tb_clk <= '0';
  while (true) loop
    wait for HALF_CLK_PERIOD;
    tb_clk <= not tb_clk;
  end loop;
  wait;
end process;

gen_reset_proc : process
begin
  tb_rst <= '1';
  wait for 10 ns;
  tb_rst <= '0';
  wait;
end process;

acknowledge_tb_end: process(tb_clk,tb_rst)
begin
  if (tb_rst = '1') then
    tb_global_ready <= '1';
    tb_stop <= '0';
  elsif rising_edge(tb_clk) then
    if (tb_global_valid = '1') then
      tb_global_ready <= '0';
      tb_stop <= '1';
    end if;
  end if;
end process;

generate_idle_signal: process(tb_clk,tb_rst)
begin
  if (tb_rst = '1') then
    tb_temp_idle <= '1';
  elsif rising_edge(tb_clk) then
    tb_temp_idle <= tb_temp_idle;
    if (start_valid = '1') then
      tb_temp_idle <= '0';
    end if;
    if(tb_stop = '1') then
      tb_temp_idle <= '1';
    end if;
  end if;
end process generate_idle_signal;

generate_start_signal : process(tb_clk, tb_rst)
begin
  if (tb_rst = '1') then
    start_valid <= '0';
    tb_started <= '0';
  elsif rising_edge(tb_clk) then
    if (tb_started = '0') then
      start_valid <= '1';
      tb_started <= '1';
    else
      start_valid <= start_valid and (not start_ready);
    end if;
  end if;
end process generate_start_signal;

transaction_increment : process
begin
  wait until tb_rst = '0';
  while (tb_temp_idle /= '1') loop
    wait until tb_clk'event and tb_clk = '1';
  end loop;
  wait until tb_temp_idle = '0';
  while (true) loop
    while (tb_temp_idle /= '1') loop
      wait until tb_clk'event and tb_clk = '1';
    end loop;
    transaction_idx := transaction_idx + 1;
    wait until tb_temp_idle = '0';
  end loop;
end process;
)DELIM";

const char *procWriteTransactions = R"DELIM(
write_output_transactor_{0}_runtime_proc : process
  file fp             : TEXT;
  variable fstatus    : FILE_OPEN_STATUS;
  variable token_line : LINE;
  variable token      : STRING(1 to 1024);
begin
  file_open(fstatus, fp, OUTPUT_{1} , WRITE_MODE);
  if (fstatus /= OPEN_OK) then
    assert false report "Open file " & OUTPUT_{2} & " failed!!!" severity note;
    assert false report "ERROR: Simulation using HLS TB failed." severity failure;
  end if;
  write(token_line, string'("[[[runtime]]]"));
  writeline(fp, token_line);
  file_close(fp);
  while transaction_idx /= TRANSACTION_NUM loop
    wait until tb_clk'event and tb_clk = '1';
  end loop;
  wait until tb_clk'event and tb_clk = '1';
  wait until tb_clk'event and tb_clk = '1';
  file_open(fstatus, fp, OUTPUT_{3}, APPEND_MODE);
  if (fstatus /= OPEN_OK) then
    assert false report "Open file " & OUTPUT_{4} & " failed!!!" severity note;
    assert false report "ERROR: Simulation using HLS TB failed." severity failure;
  end if;
  write(token_line, string'("[[[/runtime]]]"));
  writeline(fp, token_line);
  file_close(fp);
  wait;
end process;
)DELIM";

// Names of the ports of the RAM model used in the testbench (i.e., the
// dual-port RAM model).
static const string CLK_PORT = "clk";
static const string RST_PORT = "rst";
static const string CE0_PORT = "ce0";
static const string WE0_PORT = "we0";
static const string D_IN0_PORT = "din0";
static const string D_OUT0_PORT = "dout0";
static const string ADDR0_PORT = "address0";
static const string CE1_PORT = "ce1";
static const string WE1_PORT = "we1";
static const string D_IN1_PORT = "din1";
static const string D_OUT1_PORT = "dout1";
static const string ADDR1_PORT = "address1";
static const string DONE_PORT = "done";
static const string IN_FILE_PARAM = "TV_IN";
static const string OUT_FILE_PARAM = "TV_OUT";
static const string DATA_WIDTH_PARAM = "DATA_WIDTH";
static const string ADDR_WIDTH_PARAM = "ADDR_WIDTH";
static const string DATA_DEPTH_PARAM = "DEPTH";

// Start signal names
static const string START_VALID = "start_valid";
static const string START_READY = "start_ready";

// Completion signal names
static const string END_VALID = "end_valid";
static const string END_READY = "end_ready";

// Name of the return value, it is also called "out0" but it means something
// else as the D_IN0_PORT (i.e., the first port of the dual-port RAM).
static const string RET_VALUE_NAME = "out0";

// This is a helper class to generete a HDL instance
// Example:
//   Instance("my_module", "my_instance")
//     .parameter("param1", "value1")
//     .parameter("param2", "value2")
//     .connect("port1", "signal1")
//     .connect("port2", "signal2");
//   instance.emitVhdl(os);
// This will generate the following VHDL code:
//   my_instance: entity work.my_module
//     generic map(
//       param1 => value1,
//       param2 => value2
//     )
//     port map(
//       port1 => signal1,
//       port2 => signal2
//     );
class Instance {
  std::string moduleName;
  std::string instanceName;
  std::vector<std::pair<std::string, std::string>> connections;
  std::vector<std::pair<std::string, std::string>> params;

public:
  Instance(std::string module, std::string inst,
           const std::vector<unsigned> &p = {})
      : moduleName(std::move(module)), instanceName(std::move(inst)) {}

  Instance &parameter(const std::string &parameter, const std::string &value) {
    params.emplace_back(parameter, value);
    return *this;
  }

  Instance &connect(const std::string &port, const std::string &signal) {
    connections.emplace_back(port, signal);
    return *this;
  }

  void emitVhdl(mlir::raw_indented_ostream &os) {
    os << instanceName << ": entity work." << moduleName << "\n";

    // VHDL port map needs a continuous assignment
    std::sort(connections.begin(), connections.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    if (!params.empty()) {
      os << "generic map(\n";
      os.indent();
      for (size_t i = 0; i < params.size(); ++i) {
        os << params[i].first << " => " << params[i].second;
        if (i != params.size() - 1)
          os << ",\n";
      }
      os.unindent();
      os << ")\n";
    }
    os << "port map(\n";
    os.indent();
    for (size_t i = 0; i < connections.size(); ++i) {
      const auto &[port, sig] = connections[i];
      os << port << " => " << sig;
      if (i != connections.size() - 1)
        os << ",";
      os << "\n";
    }
    os.unindent();
    os << ");\n\n";
  }
};

// Declare a signal. Usage:
// - declareSTL(os, "signal_name"); declares a std_logic signal
// - declareSTL(os, "signal_name", "SIGNAL_WIDTH"); declares a std_logic_vector
void declareSTL(mlir::raw_indented_ostream &os, const string &name,
                std::optional<std::string> size = std::nullopt,
                std::optional<std::string> initialValue = std::nullopt) {
  os << "signal " << name << " : std_logic";
  if (size)
    os << "_vector(" << *size << " - 1 downto 0)";
  if (initialValue)
    os << " := " << *initialValue;
  os << ";\n";
}

void declareConstant(mlir::raw_indented_ostream &os, const string &name,
                     const string &type, const string &value) {
  os << "constant " << name << " : " << type << " := " << value << ";\n";
}

// function to get the port name in the entitiy for each paramter
void getConstantDeclaration(mlir::raw_indented_ostream &os,
                            VerificationContext &ctx) {

  handshake::FuncOp *funcOp = ctx.funcOp;

  std::string inputVectorPath = ctx.getInputVectorDir();
  std::string outputFilePath = ctx.getVhdlOutDir();

  declareConstant(os, "HALF_CLK_PERIOD", "TIME", "2.00 ns");
  declareConstant(os, "TRANSACTION_NUM", "INTEGER", to_string(1));

  for (auto [arg, portAttr] : llvm::zip_equal(
           funcOp->getBodyBlock()->getArguments(), funcOp->getArgNames())) {

    std::string argName = portAttr.dyn_cast<StringAttr>().data();

    // For scalar and array arguments we need to declare files that hold the
    // values of them
    if (isa<handshake::ChannelType, mlir::MemRefType>(arg.getType())) {
      declareConstant(os, "INPUT_" + argName, "STRING",
                      "\"" + inputVectorPath + "/input_" + argName + ".dat" +
                          "\"");

      declareConstant(os, "OUTPUT_" + argName, "STRING",
                      "\"" + outputFilePath + "/output_" + argName + ".dat" +
                          "\"");
    }

    if (handshake::ChannelType type =
            arg.getType().dyn_cast<handshake::ChannelType>()) {
      int dataWidth = type.getDataBitWidth();
      declareConstant(os, "DATA_WIDTH_" + argName, "INTEGER",
                      to_string(dataWidth));
    } else if (mlir::MemRefType type =
                   dyn_cast<mlir::MemRefType>(arg.getType())) {

      int dataWidth = type.getElementType().getIntOrFloatBitWidth();
      declareConstant(os, "DATA_WIDTH_" + argName, "INTEGER",
                      to_string(dataWidth));
      int dataDepth = type.getNumElements();
      declareConstant(os, "DATA_DEPTH_" + argName, "INTEGER",
                      to_string(dataDepth));
      int addrWidth = ((int)ceil(log2(dataDepth)));
      declareConstant(os, "ADDR_WIDTH_" + argName, "INTEGER",
                      to_string(addrWidth));
    }
  }

  for (auto [resType, portAttr] :
       llvm::zip_equal(funcOp->getResultTypes(), funcOp->getResNames())) {
    std::string argName = portAttr.dyn_cast<StringAttr>().str();
    if (handshake::ChannelType type =
            dyn_cast<handshake::ChannelType>(resType)) {
      declareConstant(os, "INPUT_" + argName, "STRING", "\"\"");
      declareConstant(os, "OUTPUT_" + argName, "STRING",
                      "\"" + outputFilePath + "/output_" + argName + ".dat" +
                          "\"");
      int dataWidth = type.getDataBitWidth();
      declareConstant(os, "DATA_WIDTH_" + argName, "INTEGER",
                      to_string(dataWidth));
    }
  }
}

// This writes the signal declarations fot the testbench
// Example:
// signal tb_clk : std_logic := '0';
// signal tb_start_value : std_logic := '0';
void getSignalDeclaration(mlir::raw_indented_ostream &os,
                          VerificationContext &ctx) {

  handshake::FuncOp *funcOp = ctx.funcOp;

  declareSTL(os, "tb_clk", std::nullopt, "'0'");
  declareSTL(os, "tb_start_value", std::nullopt, "'0'");
  declareSTL(os, "tb_rst", std::nullopt, "'0'");
  declareSTL(os, "tb_started");
  declareSTL(os, "tb_global_valid");
  declareSTL(os, "tb_global_ready");
  declareSTL(os, "tb_stop");

  for (auto [arg, portAttr] : llvm::zip_equal(
           funcOp->getBodyBlock()->getArguments(), funcOp->getArgNames())) {

    std::string argName = portAttr.dyn_cast<StringAttr>().data();

    if (handshake::ChannelType type =
            arg.getType().dyn_cast<handshake::ChannelType>()) {
      int dataWidth = type.getDataBitWidth();
      declareSTL(os, argName + "_" + CE0_PORT);
      declareSTL(os, argName + "_" + WE0_PORT);

      declareSTL(os, argName + "_dout0", to_string(dataWidth));
      declareSTL(os, argName + "_din0", to_string(dataWidth));

      declareSTL(os, argName + "_dout0" + "_valid");
      declareSTL(os, argName + "_dout0" + "_ready");

    } else if (handshake::ControlType type =
                   arg.getType().dyn_cast<handshake::ControlType>()) {
      declareSTL(os, argName + "_valid");
      declareSTL(os, argName + "_ready");
    } else if (mlir::MemRefType type =
                   dyn_cast<mlir::MemRefType>(arg.getType())) {

      int dataWidth = type.getElementType().getIntOrFloatBitWidth();
      int dataDepth = type.getNumElements();
      int addrWidth = ((int)ceil(log2(dataDepth)));

      declareSTL(os, argName + "_" + CE0_PORT);
      declareSTL(os, argName + "_" + WE0_PORT);
      declareSTL(os, argName + "_din0", to_string(dataWidth));
      declareSTL(os, argName + "_dout0", to_string(dataWidth));
      declareSTL(os, argName + "_" + ADDR0_PORT, to_string(addrWidth));

      declareSTL(os, argName + "_" + CE1_PORT);
      declareSTL(os, argName + "_" + WE1_PORT);

      declareSTL(os, argName + "_din1", to_string(dataWidth));
      declareSTL(os, argName + "_dout1", to_string(dataWidth));
      declareSTL(os, argName + "_" + ADDR1_PORT, to_string(addrWidth));
    }
  }

  for (auto [resType, portAttr] :
       llvm::zip_equal(funcOp->getResultTypes(), funcOp->getResNames())) {
    std::string argName = portAttr.dyn_cast<StringAttr>().str();
    if (handshake::ChannelType type =
            dyn_cast<handshake::ChannelType>(resType)) {
      int dataWidth = type.getDataBitWidth();
      declareSTL(os, argName + "_" + CE0_PORT);
      declareSTL(os, argName + "_" + WE0_PORT);

      declareSTL(os, argName + "_dout0", to_string(dataWidth));
      declareSTL(os, argName + "_din0", to_string(dataWidth));

      // Output channel of the single_argument block
      declareSTL(os, argName + "_dout0" + "_valid");
      declareSTL(os, argName + "_dout0" + "_ready");

      // To the input channel of the single_argument block & the tb_join
      declareSTL(os, argName + "_valid");
      declareSTL(os, argName + "_ready");
    } else if (handshake::ControlType type =
                   dyn_cast<handshake::ControlType>(resType)) {
      declareSTL(os, argName + "_valid");
      declareSTL(os, argName + "_ready");
    }
  }

  os << "\n";

  declareSTL(os, "tb_temp_idle", std::nullopt, "'1'");
  os << "shared variable transaction_idx : INTEGER := 0;\n";
  os.flush();
}

void getMemoryInstanceGeneration(mlir::raw_indented_ostream &os,
                                 VerificationContext &ctx) {

  handshake::FuncOp *funcOp = ctx.funcOp;

  for (auto [arg, portAttr] : llvm::zip_equal(
           funcOp->getBodyBlock()->getArguments(), funcOp->getArgNames())) {

    std::string argName = portAttr.dyn_cast<StringAttr>().data();

    if (handshake::ChannelType type =
            arg.getType().dyn_cast<handshake::ChannelType>()) {
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
          .connect(CE0_PORT, "'1'")
          .connect(DONE_PORT, "tb_temp_idle")
          .connect(D_OUT0_PORT, argName + "_dout0")
          .connect(D_OUT0_PORT + "_valid", argName + "_dout0_valid")
          .connect(D_OUT0_PORT + "_ready", argName + "_dout0_ready")
          .connect(WE0_PORT, "'0'")
          .connect(D_IN0_PORT, "(others => '0')");
      argInst.emitVhdl(os);

    } else if (mlir::MemRefType type =
                   dyn_cast<mlir::MemRefType>(arg.getType())) {

      Instance memInst("two_port_RAM", "mem_inst_" + argName);

      memInst.parameter(IN_FILE_PARAM, "INPUT_" + argName)
          .parameter(OUT_FILE_PARAM, "OUTPUT_" + argName)
          .parameter(DATA_WIDTH_PARAM, "DATA_WIDTH_" + argName)
          .parameter(ADDR_WIDTH_PARAM, "ADDR_WIDTH_" + argName)
          .parameter(DATA_DEPTH_PARAM, "DATA_DEPTH_" + argName)
          // NOTE: The lhs of the connect (e.g., CLK_PORT) is the port name of
          // the instantiated RAM models (e.g., two_port_RAM.vhd and
          // single_argument.vhd). This name is not necessarily named in the
          // same way as the testbench signal (e.g., tb_clk). For Dynamatic's
          // internal coverification, we have the freedom to name the interface
          // of the RAM model however we want.
          .connect(CLK_PORT, "tb_" + CLK_PORT)
          .connect(RST_PORT, "tb_" + RST_PORT)
          .connect(CE0_PORT, argName + "_" + CE0_PORT)
          .connect(WE0_PORT, argName + "_" + WE0_PORT)
          .connect(ADDR0_PORT, argName + "_" + ADDR0_PORT)
          .connect(D_OUT0_PORT, argName + "_dout0")
          .connect(D_IN0_PORT, argName + "_din0")
          .connect(CE1_PORT, argName + "_" + CE1_PORT)
          .connect(WE1_PORT, argName + "_" + WE1_PORT)
          .connect(ADDR1_PORT, argName + "_" + ADDR1_PORT)
          .connect(D_OUT1_PORT, argName + "_dout1")
          .connect(D_IN1_PORT, argName + "_din1")
          .connect(DONE_PORT, "tb_stop");

      memInst.emitVhdl(os);
    }
  }

  for (auto [resType, portAttr] :
       llvm::zip_equal(funcOp->getResultTypes(), funcOp->getResNames())) {
    std::string argName = portAttr.dyn_cast<StringAttr>().str();
    if (handshake::ChannelType type =
            dyn_cast<handshake::ChannelType>(resType)) {
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
          .connect(CE0_PORT, "'1'")
          .connect(DONE_PORT, "tb_temp_idle")
          .connect(D_OUT0_PORT, argName + "_dout0")
          .connect(D_OUT0_PORT + "_valid", argName + "_dout0_valid")
          .connect(D_OUT0_PORT + "_ready", argName + "_dout0_ready")
          .connect(WE0_PORT, argName + "_valid")
          .connect(D_IN0_PORT, argName + "_din0");
      argInst.emitVhdl(os);
    }
  }

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

    } else if (handshake::ControlType type =
                   dyn_cast<handshake::ControlType>(resType)) {
    }
  }
  joinInst.connect("outs_valid", "tb_global_valid");
  joinInst.connect("outs_ready", "tb_global_ready");

  joinInst.emitVhdl(os);
  os.flush();
}

void getDuvInstanceGeneration(mlir::raw_indented_ostream &os,
                              VerificationContext &ctx) {

  std::string duvName = ctx.vhdlDUVEntityName;

  Instance duvInst(duvName, "duv_inst");

  duvInst.connect(CLK_PORT, "tb_" + CLK_PORT)
      .connect(RST_PORT, "tb_" + RST_PORT);

  handshake::FuncOp *funcOp = ctx.funcOp;

  // Connect the input channels to the DUV
  for (auto [arg, portAttr] : llvm::zip_equal(
           funcOp->getBodyBlock()->getArguments(), funcOp->getArgNames())) {

    std::string argName = portAttr.dyn_cast<StringAttr>().data();

    if (isa<handshake::ChannelType>(arg.getType())) {
      duvInst.connect(argName, argName + "_dout0")
          .connect(argName + "_valid", argName + "_dout0_valid")
          .connect(argName + "_ready", argName + "_dout0_ready");
    } else if (handshake::ControlType type =
                   arg.getType().dyn_cast<handshake::ControlType>()) {
      // @Jiahui17: TODO: create a new argument for dataless input channels
      duvInst.connect(argName + "_valid", "\'1\'")
          .connect(argName + "_ready", argName + "_ready");
    }
  }

  // Connect the memory elements to the DUV
  for (auto [arg, portAttr] : llvm::zip_equal(
           funcOp->getBodyBlock()->getArguments(), funcOp->getArgNames())) {

    std::string argName = portAttr.dyn_cast<StringAttr>().data();

    if (isa<mlir::MemRefType>(arg.getType())) {
      duvInst.connect(argName + "_" + ADDR0_PORT, argName + "_" + ADDR0_PORT)
          .connect(argName + "_" + CE0_PORT, argName + "_" + CE0_PORT)
          .connect(argName + "_" + WE0_PORT, argName + "_" + WE0_PORT)
          .connect(argName + "_" + D_IN0_PORT, argName + "_" + D_OUT0_PORT)
          .connect(argName + "_" + D_OUT0_PORT, argName + "_" + D_IN0_PORT)
          .connect(argName + "_" + ADDR1_PORT, argName + "_" + ADDR1_PORT)
          .connect(argName + "_" + CE1_PORT, argName + "_" + CE1_PORT)
          .connect(argName + "_" + WE1_PORT, argName + "_" + WE1_PORT)
          .connect(argName + "_" + D_IN1_PORT, argName + "_" + D_OUT1_PORT)
          .connect(argName + "_" + D_OUT1_PORT, argName + "_" + D_IN1_PORT);
    }
  }

  // Connect the output channels to the DUV
  for (auto [resType, portAttr] :
       llvm::zip_equal(funcOp->getResultTypes(), funcOp->getResNames())) {
    std::string argName = portAttr.dyn_cast<StringAttr>().str();
    if (isa<handshake::ChannelType>(resType)) {
      duvInst.connect(argName, argName + "_din0")
          .connect(argName + "_valid", argName + "_valid")
          .connect(argName + "_ready", argName + "_ready");
    } else if (isa<handshake::ControlType>(resType)) {
      duvInst.connect(argName + "_valid", argName + "_valid")
          .connect(argName + "_ready", argName + "_ready");
    }
  }

  duvInst.emitVhdl(os);
}

void getOutputTagGeneration(mlir::raw_indented_ostream &os,
                            VerificationContext &ctx) {
  handshake::FuncOp *funcOp = ctx.funcOp;

  // Connect the memory elements to the DUV
  for (auto [arg, portAttr] : llvm::zip_equal(
           funcOp->getBodyBlock()->getArguments(), funcOp->getArgNames())) {

    std::string argName = portAttr.dyn_cast<StringAttr>().data();

    if (isa<mlir::MemRefType, handshake::ChannelType>(arg.getType())) {
      // os << "------------------------------------------------------------\n";
      // os << "-- Write [[[runtime]]], [[[/runtime]]] for output transactor\n";
      os << llvm::formatv(procWriteTransactions, argName, argName, argName,
                          argName, argName);
    }
  }

  // Connect the output channels to the DUV
  for (auto [resType, portAttr] :
       llvm::zip_equal(funcOp->getResultTypes(), funcOp->getResNames())) {
    std::string argName = portAttr.dyn_cast<StringAttr>().str();
    if (isa<handshake::ChannelType>(resType)) {
      // os << "------------------------------------------------------------\n";
      // os << "-- Write [[[runtime]]], [[[/runtime]]] for output transactor\n";
      os << llvm::formatv(procWriteTransactions, argName, argName, argName,
                          argName, argName);
    }
  }
}

void vhdlTbCodegen(VerificationContext &ctx) {

  std::error_code ec;
  llvm::raw_fd_ostream fileStream(ctx.getVhdlTestbenchPath(), ec);
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
  getOutputTagGeneration(os, ctx);
  os << COMMON_TB_BODY;
  os.unindent();
  os << "end architecture behavior;\n";
  os.flush();
}

} // namespace hls_verify
