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

// class Constant

Constant::Constant(const string &name, const string &type, const string &value)
    : constName(name), constType(type), constValue(value) {}

// class MemElem

static const string CLK_PORT = "clk";
static const string RST_PORT = "rst";
static const string CE0_PORT = "ce0";
static const string WE0_PORT = "we0";
static const string D_IN0_PORT = "mem_din0";
static const string D_OUT0_PORT = "mem_dout0";
static const string ADDR0_PORT = "address0";
static const string CE1_PORT = "ce1";
static const string WE1_PORT = "we1";
static const string D_IN1_PORT = "mem_din1";
static const string D_OUT1_PORT = "mem_dout1";
static const string ADDR1_PORT = "address1";
static const string DONE_PORT = "done";
static const string IN_FILE_PARAM = "TV_IN";
static const string OUT_FILE_PARAM = "TV_OUT";
static const string DATA_WIDTH_PARAM = "DATA_WIDTH";
static const string ADDR_WIDTH_PARAM = "ADDR_WIDTH";
static const string DATA_DEPTH_PARAM = "DEPTH";

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

string HlsVhdlTb::getAddr0PortNameForCParam(string &cParam) {
  return cParam + "_address0";
}

string HlsVhdlTb::getAddr1PortNameForCParam(string &cParam) {
  return cParam + "_address1";
}

string HlsVhdlTb::getCe0PortNameForCParam(string &cParam) {
  return cParam + "_ce0";
}

string HlsVhdlTb::getCe1PortNameForCParam(string &cParam) {
  return cParam + "_ce1";
}

string HlsVhdlTb::getDataIn0PortNameForCParam(string &cParam) {
  return cParam + "_din0";
}

string HlsVhdlTb::getDataIn1PortNameForCParam(string &cParam) {
  return cParam + "_din1";
}

string HlsVhdlTb::getDataOut0PortNameForCParam(string &cParam) {
  return cParam + "_dout0";
}

string HlsVhdlTb::getDataOut1PortNameForCParam(string &cParam) {
  return cParam + "_dout1";
}

string HlsVhdlTb::getWe0PortNameForCParam(string &cParam) {
  return cParam + "_we0";
}

string HlsVhdlTb::getWe1PortNameForCParam(string &cParam) {
  return cParam + "_we1";
}

string HlsVhdlTb::getValidInPortNameForCParam(string &cParam) {
  return cParam + "_valid_in";
}

string HlsVhdlTb::getValidOutPortNameForCParam(string &cParam) {
  return cParam + "_valid_out";
}

string HlsVhdlTb::getReadyInPortNameForCParam(string &cParam) {
  return cParam + "_ready_in";
}

string HlsVhdlTb::getReadyOutPortNameForCParam(string &cParam) {
  return cParam + "_ready_out";
}

string HlsVhdlTb::getDataInSaPortNameForCParam(string &cParam) {
  return cParam + "_din";
}

string HlsVhdlTb::getDataOutSaPortNameForCParam(string &cParam) {
  return cParam + "_dout";
}

void HlsVhdlTb::getLibraryHeader(mlir::raw_indented_ostream &os) {
  os << VHDL_LIBRARY_HEADER;
}

void HlsVhdlTb::getEntitiyDeclaration(mlir::raw_indented_ostream &os) {
  os << "entity " + tleName + " is\n\nend entity " + tleName + ";\n";
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

void HlsVhdlTb::getArchitectureBegin(mlir::raw_indented_ostream &os) {
  os << "architecture behav of " << tleName << " is\n\n";
  os.indent();
  os << "-- Constant declarations\n";
  getConstantDeclaration(os);
  os << "-- Signal declarations\n\n";
  getSignalDeclaration(os);
  os.unindent();
  os << "begin\n\n";
}

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
  Instance(std::string module, std::string inst, std::vector<unsigned> p = {})
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
  declareSTL(os, "tb_start_valid", std::nullopt, "'0'");
  declareSTL(os, "tb_start_ready");
  declareSTL(os, "tb_started");
  declareSTL(os, "tb_end_valid");
  declareSTL(os, "tb_end_ready");
  declareSTL(os, "tb_out0_valid");
  declareSTL(os, "tb_out0_ready");
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
          .connect(CLK_PORT, "tb_clk")
          .connect(RST_PORT, "tb_rst")
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
          .connect(CLK_PORT, "tb_clk")
          .connect(RST_PORT, "tb_rst")
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
        argInst.connect(WE0_PORT, "tb_out0_valid")
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
    joinInst.connect("ins_valid(" + std::to_string(idx) + ")", "tb_out0_valid");
    joinInst.connect("ins_ready(" + std::to_string(idx++) + ")",
                     "tb_out0_ready");
  }
  for (MemElem &m : memElems) {
    if (m.isArray) {
      joinInst.connect("ins_valid(" + std::to_string(idx) + ")",
                       m.memEndSignalName + "_valid");
      joinInst.connect("ins_ready(" + std::to_string(idx++) + ")",
                       m.memEndSignalName + "_ready");
    }
  }
  joinInst.connect("ins_valid(" + std::to_string(idx) + ")", "tb_end_valid");
  joinInst.connect("ins_ready(" + std::to_string(idx++) + ")", "tb_end_ready");
  joinInst.connect("outs_valid", "tb_global_valid");
  joinInst.connect("outs_ready", "tb_global_ready");

  joinInst.emitVhdl(os);
}

void HlsVhdlTb::getDuvInstanceGeneration(mlir::raw_indented_ostream &os) {
  duvPortMap.emplace_back("clk", "tb_clk");
  duvPortMap.emplace_back("rst", "tb_rst");

  for (size_t i = 0; i < cDuvParams.size(); i++) {
    CFunctionParameter p = cDuvParams[i];
    MemElem m = memElems[i];

    if (m.isArray) {
      duvPortMap.emplace_back(getAddr0PortNameForCParam(p.parameterName),
                              m.addr0SignalName);
      duvPortMap.emplace_back(getCe0PortNameForCParam(p.parameterName),
                              m.ce0SignalName);
      duvPortMap.emplace_back(getWe0PortNameForCParam(p.parameterName),
                              m.we0SignalName);
      duvPortMap.emplace_back(getDataIn0PortNameForCParam(p.parameterName),
                              m.dOut0SignalName);
      duvPortMap.emplace_back(getDataOut0PortNameForCParam(p.parameterName),
                              m.dIn0SignalName);

      duvPortMap.emplace_back(getAddr1PortNameForCParam(p.parameterName),
                              m.addr1SignalName);
      duvPortMap.emplace_back(getCe1PortNameForCParam(p.parameterName),
                              m.ce1SignalName);
      duvPortMap.emplace_back(getWe1PortNameForCParam(p.parameterName),
                              m.we1SignalName);
      duvPortMap.emplace_back(getDataIn1PortNameForCParam(p.parameterName),
                              m.dOut1SignalName);
      duvPortMap.emplace_back(getDataOut1PortNameForCParam(p.parameterName),
                              m.dIn1SignalName);

      // Memory start signal
      duvPortMap.emplace_back(p.parameterName + "_start_valid", "'1'");
      duvPortMap.emplace_back(p.parameterName + "_start_ready",
                              m.memStartSignalName + "_ready");

      // Memory completion signal
      duvPortMap.emplace_back(p.parameterName + "_end_valid",
                              m.memEndSignalName + "_valid");
      duvPortMap.emplace_back(p.parameterName + "_end_ready",
                              m.memEndSignalName + "_ready");

    } else {
      if (p.isInput) {
        duvPortMap.emplace_back(p.parameterName, m.dOut0SignalName);
        duvPortMap.emplace_back(p.parameterName + "_valid",
                                m.dOut0SignalName + "_valid");
        duvPortMap.emplace_back(p.parameterName + "_ready",
                                m.dOut0SignalName + "_ready");
      }

      if (p.isOutput) {
        if (p.isReturn) {
          duvPortMap.emplace_back(p.parameterName, m.dIn0SignalName);
          duvPortMap.emplace_back(p.parameterName + "_valid", "tb_out0_valid");
          duvPortMap.emplace_back(p.parameterName + "_ready", "tb_out0_ready");
        } else {
          duvPortMap.emplace_back(getValidOutPortNameForCParam(p.parameterName),
                                  m.we0SignalName);
          duvPortMap.emplace_back(
              getDataOutSaPortNameForCParam(p.parameterName), m.dIn0SignalName);
          duvPortMap.emplace_back(getReadyInPortNameForCParam(p.parameterName),
                                  "'1'");
        }
      }
    }
  }

  // Start and end signal
  duvPortMap.emplace_back("start_valid", "tb_start_valid");
  duvPortMap.emplace_back("start_ready", "tb_start_ready");
  duvPortMap.emplace_back("end_valid", "tb_end_valid");
  duvPortMap.emplace_back("end_ready", "tb_end_ready");

  os << "duv: entity work." << duvName << "\n";
  os << "port map (\n";
  os.indent();
  for (size_t i = 0; i < duvPortMap.size(); i++) {
    pair<string, string> elem = duvPortMap[i];
    os << "" << elem.first << " => " << elem.second
       << ((i < duvPortMap.size() - 1) ? "," : "") << "\n";
  }
  os.unindent();
  os << ");\n\n";
}

void HlsVhdlTb::getCommonBody(mlir::raw_indented_ostream &os) {
  os << COMMON_TB_BODY;
}

void HlsVhdlTb::getArchitectureEnd(mlir::raw_indented_ostream &os) {
  os << "end architecture behav;\n";
}

void HlsVhdlTb::getOutputTagGeneration(mlir::raw_indented_ostream &os) {
  os << "-------------------------------------------------------------------\n";
  os << "-- Write \"[[[runtime]]]\" and \"[[[/runtime]]]\" for output "
        "transactor\n";
  for (auto &cDuvParam : cDuvParams) {
    if (cDuvParam.isOutput) {
      os << "write_output_transactor_" << cDuvParam.parameterName
         << "_runtime_proc : process\n";
      os << "file fp             : TEXT;\n";
      os << "variable fstatus    : FILE_OPEN_STATUS;\n";
      os << "variable token_line : LINE;\n";
      os << "variable token      : STRING(1 to 1024);\n";
      os << "\n";
      os << "begin\n";
      os.indent();
      os << "file_open(fstatus, fp, OUTPUT_" << cDuvParam.parameterName
         << ", WRITE_MODE);\n";
      os << "if (fstatus /= OPEN_OK) then\n";
      os << "assert false report \"Open file \" & OUTPUT_"
         << cDuvParam.parameterName << " & \" failed!!!\" severity note;\n";
      os << "assert false report \"ERROR: Simulation using HLS TB "
            "failed.\" severity failure;\n";
      os << "end if;\n";
      os << "write(token_line, string'(\"[[[runtime]]]\"));\n";
      os << "writeline(fp, token_line);\n";
      os << "file_close(fp);\n";
      os << "while transaction_idx /= TRANSACTION_NUM loop\n";
      os << "wait until tb_clk'event and tb_clk = '1';\n";
      os << "end loop;\n";
      os << "wait until tb_clk'event and tb_clk = '1';\n";
      os << "wait until tb_clk'event and tb_clk = '1';\n";
      os << "file_open(fstatus, fp, OUTPUT_" << cDuvParam.parameterName
         << ", APPEND_MODE);\n";
      os << "if (fstatus /= OPEN_OK) then\n";
      os << "assert false report \"Open file \" & OUTPUT_"
         << cDuvParam.parameterName << " & \" failed!!!\" severity note;\n";
      os << "assert false report \"ERROR: Simulation using HLS TB "
            "failed.\" severity failure;\n";
      os << "end if;\n";
      os << "write(token_line, string'(\"[[[/runtime]]]\"));\n";
      os << "writeline(fp, token_line);\n";
      os << "file_close(fp);\n";
      os << "wait;\n";
      os.unindent();
      os << "end process;\n";
    }
  }
  os << "-----------------------------------------------------------------\n\n";
}

void HlsVhdlTb::generateVhdlTestbench(mlir::raw_indented_ostream &os) {
  getLibraryHeader(os);
  getEntitiyDeclaration(os);
  getArchitectureBegin(os);
  os.indent();
  getDuvInstanceGeneration(os);
  getMemoryInstanceGeneration(os);
  getOutputTagGeneration(os);
  getCommonBody(os);
  os.unindent();
  getArchitectureEnd(os);
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
