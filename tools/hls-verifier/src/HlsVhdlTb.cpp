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
#include "mlir/Support/IndentedOstream.h"
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
----------------------------------------------------------------------------
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
----------------------------------------------------------------------------
gen_clock_proc : process
begin
  tb_clk <= '0';
  while (true) loop
    wait for HALF_CLK_PERIOD;
    tb_clk <= not tb_clk;
  end loop;
  wait;
end process;
----------------------------------------------------------------------------
gen_reset_proc : process
begin
  tb_rst <= '1';
  wait for 10 ns;
  tb_rst <= '0';
  wait;
end process;
----------------------------------------------------------------------------
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
----------------------------------------------------------------------------
generate_idle_signal: process(tb_clk,tb_rst)
begin
  if (tb_rst = '1') then
    tb_temp_idle <= '1';
  elsif rising_edge(tb_clk) then
    tb_temp_idle <= tb_temp_idle;
    if (tb_start_valid = '1') then
      tb_temp_idle <= '0';
    end if;
    if(tb_stop = '1') then
      tb_temp_idle <= '1';
    end if;
  end if;
end process generate_idle_signal;
----------------------------------------------------------------------------
generate_start_signal : process(tb_clk, tb_rst)
begin
  if (tb_rst = '1') then
    tb_start_valid <= '0';
    tb_started <= '0';
  elsif rising_edge(tb_clk) then
    if (tb_started = '0') then
      tb_start_valid <= '1';
      tb_started <= '1';
    else
      tb_start_valid <= tb_start_valid and (not tb_start_ready);
    end if;
  end if;
end process generate_start_signal;
----------------------------------------------------------------------------
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
--------------------------------------------------------------------------
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
    assert false
    report "Open file " & OUTPUT_{2} & " failed!!!"
    severity note;
    assert false
    report "ERROR: Simulation using HLS TB failed."
    severity failure;
  end if;
  write(token_line, string'("[[[runtime]]]"));
  writeline(fp, token_line);
  file_close(fp);
  while transaction_idx /= TRANSACTION_NUM loop
    wait until tb_clk'event and tb_clk = '1';
  end loop;
  wait until tb_clk'event and tb_clk = '1';
  wait until tb_clk'event and tb_clk = '1';
  file_open(fstatus, fp, OUTPUT_{3} , APPEND_MODE);
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

// class Constant

Constant::Constant(const string &name, const string &type, const string &value)
    : constName(name), constType(type), constValue(value) {}

// class MemElem

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

// class HlsVhdTb

HlsVhdlTb::HlsVhdlTb(const VerificationContext &ctx) : ctx(ctx) {
  duvName = ctx.getVhdlDuvEntityName();
  tleName = ctx.getVhdlDuvEntityName() + "_tb";
  cDuvParams = ctx.getFuvParams();

  Constant hcp("HALF_CLK_PERIOD", "TIME", "2.00 ns");
  constants.push_back(hcp);

  int transNum = getTransactionNumberFromInput();
  logInf(LOG_TAG, "Transaction number computed : " + to_string(transNum));
  if (transNum <= 0)
    logErr(LOG_TAG, "Invalid number of transactions detected!");

  Constant tN("TRANSACTION_NUM", "INTEGER", to_string(transNum));
  constants.push_back(tN);

  for (const auto &p : cDuvParams) {
    MemElem mElem;

    Constant inFileName("INPUT_" + p.parameterName, "STRING",
                        "\"" + getInputFilepathForParam(p) + "\"");
    constants.push_back(inFileName);
    mElem.inFileParamValue = inFileName.constName;

    Constant outFileName("OUTPUT_" + p.parameterName, "STRING",
                         "\"" + getOutputFilepathForParam(p) + "\"");
    constants.push_back(outFileName);
    mElem.outFileParamValue = outFileName.constName;

    Constant dataWidth("DATA_WIDTH_" + p.parameterName, "INTEGER",
                       to_string(p.dtWidth));
    constants.push_back(dataWidth);
    mElem.dataWidthParamValue = dataWidth.constName;

    mElem.isArray = (p.isPointer && p.arrayLength > 1);

    if (mElem.isArray) {

      string addrwidthvalue = to_string(((int)ceil(log2(p.arrayLength))));
      Constant addrWidth("ADDR_WIDTH_" + p.parameterName, "INTEGER",
                         addrwidthvalue);
      // to_string(((int) ceil(log2(p.arrayLength)))));
      constants.push_back(addrWidth);
      mElem.addrWidthParamValue = addrWidth.constName;
      Constant dataDepth("DATA_DEPTH_" + p.parameterName, "INTEGER",
                         to_string(p.arrayLength));
      constants.push_back(dataDepth);
      mElem.dataDepthParamValue = dataDepth.constName;
    }

    mElem.dIn0SignalName = p.parameterName + "_mem_din0";
    mElem.dOut0SignalName = p.parameterName + "_mem_dout0";
    mElem.we0SignalName = p.parameterName + "_mem_" + WE0_PORT;
    mElem.ce0SignalName = p.parameterName + "_mem_" + CE0_PORT;
    mElem.addr0SignalName = p.parameterName + "_mem_" + ADDR0_PORT;
    mElem.dIn1SignalName = p.parameterName + "_mem_din1";
    mElem.dOut1SignalName = p.parameterName + "_mem_dout1";
    mElem.we1SignalName = p.parameterName + "_mem_" + WE1_PORT;
    mElem.ce1SignalName = p.parameterName + "_mem_" + CE1_PORT;
    mElem.addr1SignalName = p.parameterName + "_mem_" + ADDR1_PORT;

    mElem.memStartSignalName = p.parameterName + "_memStart";
    mElem.memEndSignalName = p.parameterName + "_memEnd";

    memElems.push_back(mElem);
  }
}

string HlsVhdlTb::getInputFilepathForParam(const CFunctionParameter &param) {
  if (param.isInput)
    return ctx.getInputVectorPath(param);

  return "";
}

string HlsVhdlTb::getOutputFilepathForParam(const CFunctionParameter &param) {
  if (param.isOutput)
    return ctx.getVhdlOutPath(param);

  return "";
}

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

// function to get the port name in the entitiy for each paramter

void HlsVhdlTb::getConstantDeclaration(mlir::raw_indented_ostream &os) {
  for (const auto &c : constants) {
    os << "constant " << c.constName << " : " << c.constType
       << " := " << c.constValue << ";\n";
  }
}

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
    os << ");\n";
  }
};

// This writes the signal declarations fot the testbench
// Example:
// signal tb_clk : std_logic := '0';
// signal tb_start_value : std_logic := '0';
void HlsVhdlTb::getSignalDeclaration(mlir::raw_indented_ostream &os) {

  declareSTL(os, "tb_clk", std::nullopt, "'0'");
  declareSTL(os, "tb_start_value", std::nullopt, "'0'");
  declareSTL(os, "tb_rst", std::nullopt, "'0'");
  declareSTL(os, "tb_" + START_VALID, std::nullopt, "'0'");
  declareSTL(os, "tb_" + START_READY);
  declareSTL(os, "tb_started");
  declareSTL(os, "tb_" + END_VALID);
  declareSTL(os, "tb_" + END_READY);
  declareSTL(os, "tb_" + RET_VALUE_NAME + "_valid");
  declareSTL(os, "tb_" + RET_VALUE_NAME + "_ready");
  declareSTL(os, "tb_global_valid");
  declareSTL(os, "tb_global_ready");
  declareSTL(os, "tb_stop");

  for (size_t i = 0; i < cDuvParams.size(); i++) {
    CFunctionParameter p = cDuvParams[i];
    MemElem m = memElems[i];

    if (m.isArray) {

      declareSTL(os, m.ce0SignalName);
      declareSTL(os, m.we0SignalName);
      declareSTL(os, m.dIn0SignalName, m.dataWidthParamValue);
      declareSTL(os, m.dOut0SignalName, m.dataWidthParamValue);
      declareSTL(os, m.addr0SignalName, m.addrWidthParamValue);

      declareSTL(os, m.ce1SignalName);
      declareSTL(os, m.we1SignalName);

      declareSTL(os, m.dIn1SignalName, m.dataWidthParamValue);
      declareSTL(os, m.dOut1SignalName, m.dataWidthParamValue);
      declareSTL(os, m.addr1SignalName, m.addrWidthParamValue);

      declareSTL(os, m.memStartSignalName + "_valid");
      declareSTL(os, m.memStartSignalName + "_ready");
      declareSTL(os, m.memEndSignalName + "_valid");
      declareSTL(os, m.memEndSignalName + "_ready");

    } else {
      if ((cDuvParams[i].isReturn && cDuvParams[i].isOutput) ||
          !cDuvParams[i].isReturn) {

        declareSTL(os, m.ce0SignalName);
        declareSTL(os, m.we0SignalName);

        declareSTL(os, m.dOut0SignalName, m.dataWidthParamValue);
        declareSTL(os, m.dIn0SignalName, m.dataWidthParamValue);

        declareSTL(os, m.dOut0SignalName + "_valid");
        declareSTL(os, m.dOut0SignalName + "_ready");
      }
    }
  }

  os << "\n";

  declareSTL(os, "tb_temp_idle", std::nullopt, "'1'");
  os << "shared variable transaction_idx : INTEGER := 0;\n";
}

void HlsVhdlTb::getMemoryInstanceGeneration(mlir::raw_indented_ostream &os) {
  for (size_t i = 0; i < memElems.size(); i++) {
    MemElem m = memElems[i];
    CFunctionParameter p = cDuvParams[i];

    if (m.isArray) {
      // Models for array arguments

      Instance memInst("two_port_RAM", "mem_inst_" + p.parameterName);

      memInst.parameter(IN_FILE_PARAM, m.inFileParamValue)
          .parameter(OUT_FILE_PARAM, m.outFileParamValue)
          .parameter(DATA_WIDTH_PARAM, m.dataWidthParamValue)
          .parameter(ADDR_WIDTH_PARAM, m.addrWidthParamValue)
          .parameter(DATA_DEPTH_PARAM, m.dataDepthParamValue)
          .connect(CLK_PORT, "tb_" + CLK_PORT)
          .connect(RST_PORT, "tb_" + RST_PORT)
          .connect(CE0_PORT, m.ce0SignalName)
          .connect(WE0_PORT, m.we0SignalName)
          .connect(ADDR0_PORT, m.addr0SignalName)
          .connect(D_OUT0_PORT, m.dOut0SignalName)
          .connect(D_IN0_PORT, m.dIn0SignalName)
          .connect(CE1_PORT, m.ce1SignalName)
          .connect(WE1_PORT, m.we1SignalName)
          .connect(ADDR1_PORT, m.addr1SignalName)
          .connect(D_OUT1_PORT, m.dOut1SignalName)
          .connect(D_IN1_PORT, m.dIn1SignalName)
          .connect(DONE_PORT, "tb_stop");

      memInst.emitVhdl(os);

    } else {

      Instance argInst("single_argument", "arg_inst_" + p.parameterName);
      argInst.parameter(IN_FILE_PARAM, m.inFileParamValue)
          .parameter(OUT_FILE_PARAM, m.outFileParamValue)
          .parameter(DATA_WIDTH_PARAM, m.dataWidthParamValue)
          .connect(CLK_PORT, "tb_" + CLK_PORT)
          .connect(RST_PORT, "tb_" + RST_PORT)
          .connect(CE0_PORT, "'1'")
          .connect(DONE_PORT, "tb_temp_idle")
          .connect(D_OUT0_PORT, m.dOut0SignalName)
          .connect(D_OUT0_PORT + "_valid", m.dOut0SignalName + "_valid")
          .connect(D_OUT0_PORT + "_ready", m.dOut0SignalName + "_ready");

      if (p.isInput && !p.isOutput && !p.isReturn) {
        // An argument that is a pure input (e.g., in_int_t).
        argInst.connect(WE0_PORT, "'0'").connect(D_IN0_PORT, "(others => '0')");
      } else if (p.isOutput && !p.isReturn) {
        // An argument that is an output (e.g., inout_int_t or out_int_t).
        argInst.connect(WE0_PORT, m.we0SignalName)
            .connect(D_IN0_PORT, m.dIn0SignalName);
      } else if (!p.isInput && p.isOutput && p.isReturn) {
        // Return value of the function.
        argInst.connect(WE0_PORT, "tb_" + RET_VALUE_NAME + "_valid")
            .connect(D_IN0_PORT, m.dIn0SignalName);
      } else {
        assert(false && "Invalid parameter type");
      }

      argInst.emitVhdl(os);
    }

    os << "\n\n";
  }

  // Join all output valid signals into a global simulation end signal to
  // stop the simulation
  unsigned joinSize = 1 + std::count_if(memElems.begin(), memElems.end(),
                                        [](MemElem &m) { return m.isArray; });
  bool hasReturnVal = std::any_of(
      cDuvParams.begin(), cDuvParams.end(),
      [](CFunctionParameter &p) { return p.isOutput && p.isReturn; });
  if (hasReturnVal)
    joinSize += 1;

  unsigned idx = 0;

  Instance joinInst("tb_join", "join_valids");

  joinInst.parameter("SIZE", std::to_string(joinSize));

  if (hasReturnVal) {
    joinInst.connect("ins_valid(" + std::to_string(idx) + ")",
                     "tb_" + RET_VALUE_NAME + "_valid");
    joinInst.connect("ins_ready(" + std::to_string(idx++) + ")",
                     "tb_" + RET_VALUE_NAME + "_ready");
  }
  for (MemElem &m : memElems) {
    if (m.isArray) {
      joinInst.connect("ins_valid(" + std::to_string(idx) + ")",
                       m.memEndSignalName + "_valid");
      joinInst.connect("ins_ready(" + std::to_string(idx++) + ")",
                       m.memEndSignalName + "_ready");
    }
  }
  joinInst.connect("ins_valid(" + std::to_string(idx) + ")", "tb_" + END_VALID);
  joinInst.connect("ins_ready(" + std::to_string(idx++) + ")",
                   "tb_" + END_READY);
  joinInst.connect("outs_valid", "tb_global_valid");
  joinInst.connect("outs_ready", "tb_global_ready");

  joinInst.emitVhdl(os);
}

void HlsVhdlTb::getDuvInstanceGeneration(mlir::raw_indented_ostream &os) {

  Instance duvInst(duvName, "duv_inst");

  duvInst.connect(CLK_PORT, "tb_" + CLK_PORT)
      .connect(RST_PORT, "tb_" + RST_PORT)
      .connect(START_VALID, "tb_" + START_VALID)
      .connect(START_READY, "tb_" + START_READY)
      .connect(END_VALID, "tb_" + END_VALID)
      .connect(END_READY, "tb_" + END_READY);

  for (size_t i = 0; i < cDuvParams.size(); i++) {
    CFunctionParameter p = cDuvParams[i];
    MemElem m = memElems[i];

    if (m.isArray) {
      duvInst.connect(p.parameterName + "_" + ADDR0_PORT, m.addr0SignalName)
          .connect(p.parameterName + "_" + CE0_PORT, m.ce0SignalName)
          .connect(p.parameterName + "_" + WE0_PORT, m.we0SignalName)
          .connect(p.parameterName + "_" + D_IN0_PORT, m.dOut0SignalName)
          .connect(p.parameterName + "_" + D_OUT0_PORT, m.dIn0SignalName)
          .connect(p.parameterName + "_" + ADDR1_PORT, m.addr1SignalName)
          .connect(p.parameterName + "_" + CE1_PORT, m.ce1SignalName)
          .connect(p.parameterName + "_" + WE1_PORT, m.we1SignalName)
          .connect(p.parameterName + "_" + D_IN1_PORT, m.dOut1SignalName)
          .connect(p.parameterName + "_" + D_OUT1_PORT, m.dIn1SignalName)

          // Memory start signal
          .connect(p.parameterName + "_" + START_VALID, "'1'")
          .connect(p.parameterName + "_" + START_READY,
                   m.memStartSignalName + "_ready")

          // Memory completion signal
          .connect(p.parameterName + "_" + END_VALID,
                   m.memEndSignalName + "_valid")
          .connect(p.parameterName + "_" + END_READY,
                   m.memEndSignalName + "_ready");

    } else {
      if (p.isInput) {
        duvInst.connect(p.parameterName, m.dOut0SignalName)
            .connect(p.parameterName + "_valid", m.dOut0SignalName + "_valid")
            .connect(p.parameterName + "_ready", m.dOut0SignalName + "_ready");
      }

      if (p.isOutput) {
        if (p.isReturn) {
          duvInst.connect(p.parameterName, m.dIn0SignalName)
              .connect(p.parameterName + "_valid",
                       "tb_" + RET_VALUE_NAME + "_valid")
              .connect(p.parameterName + "_ready",
                       "tb_" + RET_VALUE_NAME + "_ready");
        } else {
          duvInst.connect(p.parameterName + "_valid", m.we0SignalName)
              .connect(p.parameterName + "_din", m.dIn0SignalName)
              .connect(p.parameterName + "_ready", "'1'");
        }
      }
    }
  }

  duvInst.emitVhdl(os);
}

void HlsVhdlTb::getOutputTagGeneration(mlir::raw_indented_ostream &os) {
  os << "------------------------------------------------------------\n";
  os << "-- Write [[[runtime]]], [[[/runtime]]] for output transactor\n";
  for (auto &cDuvParam : cDuvParams) {
    if (cDuvParam.isOutput) {

      os << llvm::formatv(procWriteTransactions, cDuvParam.parameterName,
                          cDuvParam.parameterName, cDuvParam.parameterName,
                          cDuvParam.parameterName, cDuvParam.parameterName);
    }
  }
  os << "-----------------------------------------------------------------\n\n";
}

void HlsVhdlTb::generateVhdlTestbench(mlir::raw_indented_ostream &os) {
  os << VHDL_LIBRARY_HEADER;
  os << "entity " + tleName + " is\nend entity " + tleName + ";\n\n";
  os << "architecture behavior of " << tleName << " is\n\n";
  os.indent();
  getConstantDeclaration(os);
  getSignalDeclaration(os);
  os.unindent();
  os << "begin\n\n";
  os.indent();
  getDuvInstanceGeneration(os);
  getMemoryInstanceGeneration(os);
  getOutputTagGeneration(os);
  os << COMMON_TB_BODY;
  os.unindent();
  os << "end architecture behavior;\n";
}

int HlsVhdlTb::getTransactionNumberFromInput() {
  bool hasInput = false;
  for (auto &cDuvParam : cDuvParams) {
    if (cDuvParam.isInput) {
      hasInput = true;
      string inputPath = getInputFilepathForParam(cDuvParam);
      return getNumberOfTransactions(inputPath);
    }
  }
  return hasInput ? -1 : 1;
}

} // namespace hls_verify
