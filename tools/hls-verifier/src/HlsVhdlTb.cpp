//===- HlsVhdlTb.cpp --------------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <sstream>

#include "HlsLogging.h"
#include "HlsVhdlTb.h"

namespace hls_verify {
const string LOG_TAG = "VVER";

string vhdlLibraryHeader = "library IEEE;\n"
                           "use ieee.std_logic_1164.all;\n"
                           "use ieee.std_logic_arith.all;\n"
                           "use ieee.std_logic_unsigned.all;\n"
                           "use ieee.std_logic_textio.all;\n"
                           "use ieee.numeric_std.all;\n"
                           "use std.textio.all;\n\n"
                           "use work.sim_package.all;\n"
                           "\n\n";

string commonTbBody =
    "\n"
    "--------------------------------------------------------------------------"
    "--\n"
    "generate_sim_done_proc : process\n"
    "begin\n"
    "	while (transaction_idx /= TRANSACTION_NUM) loop\n"
    "		wait until tb_clk'event and tb_clk = '1';\n"
    "	end loop;\n"
    "	wait until tb_clk'event and tb_clk = '1';\n"
    "	wait until tb_clk'event and tb_clk = '1';\n"
    "	wait until tb_clk'event and tb_clk = '1';\n"
    "	assert false report \"simulation done!\" severity note;\n"
    "	assert false report \"NORMAL EXIT (note: failure is to force the "
    "simulator to stop)\" severity failure;\n"
    "	wait;\n"
    "end process;\n"
    "\n"
    "--------------------------------------------------------------------------"
    "--\n"
    "gen_clock_proc : process\n"
    "begin\n"
    "	tb_clk <= '0';\n"
    "	while (true) loop\n"
    "		wait for HALF_CLK_PERIOD;\n"
    "		tb_clk <= not tb_clk;\n"
    "	end loop;\n"
    "	wait;\n"
    "end process;\n"
    "\n"
    "--------------------------------------------------------------------------"
    "--\n"
    "gen_reset_proc : process\n"
    "begin\n"
    "	tb_rst <= '1';\n"
    "	wait for 10 ns;\n"
    "	tb_rst <= '0';\n"
    "	wait;\n"
    "end process;\n"
    "\n"
    "--------------------------------------------------------------------------"
    "--\n"

    "generate_idle_signal: process(tb_clk,tb_rst)\n"
    "begin\n"
    "   if (tb_rst = '1') then\n"
    "       tb_temp_idle <= '1';\n"
    "   elsif rising_edge(tb_clk) then\n"
    "       tb_temp_idle <= tb_temp_idle;\n"
    "       if (tb_start_valid = '1') then\n"
    "           tb_temp_idle <= '0';\n"
    "       elsif(tb_end_valid = '1') then\n"
    "           tb_temp_idle <= '1';\n"
    "       end if;\n"
    "   end if;\n"
    "end process generate_idle_signal;\n"
    "\n"
    "--------------------------------------------------------------------------"
    "--\n"
    "generate_start_signal : process(tb_clk, tb_rst)\n"
    "begin\n"
    "   if (tb_rst = '1') then\n"
    "       tb_start_valid <= '0';\n"
    "       tb_started <= '0';\n"
    "   elsif rising_edge(tb_clk) then\n"
    "       if (tb_started = '0') then \n"
    "           tb_start_valid <= '1';\n"
    "           tb_started <= '1';\n"
    "       else\n"
    "           tb_start_valid <= tb_start_valid and (not tb_start_ready);\n"
    "       end if;\n"
    "   end if;\n"
    "end process generate_start_signal;\n"
    "\n"
    "--------------------------------------------------------------------------"
    "--\n"
    "transaction_increment : process\n"
    "begin\n"
    "	wait until tb_rst = '0';\n"
    "	while (tb_temp_idle /= '1') loop\n"
    "		wait until tb_clk'event and tb_clk = '1';\n"
    "	end loop;\n"
    "	wait until tb_temp_idle = '0';\n"
    "\n"
    "	while (true) loop\n"
    "		while (tb_temp_idle /= '1') loop\n"
    "			wait until tb_clk'event and tb_clk = '1';\n"
    "		end loop;\n"
    "		transaction_idx := transaction_idx + 1;\n"
    "		wait until tb_temp_idle = '0';\n"
    "	end loop;\n"
    "end process;\n"
    "\n"
    "--------------------------------------------------------------------------"
    "--\n\n";

// class Constant

Constant::Constant(const string &name, const string &type, const string &value)
    : constName(name), constType(type), constValue(value) {}

// class MemElem

string MemElem::clkPortName = "clk";
string MemElem::rstPortName = "rst";
string MemElem::ce0PortName = "ce0";
string MemElem::we0PortName = "we0";
string MemElem::dIn0PortName = "mem_din0";
string MemElem::dOut0PortName = "mem_dout0";
string MemElem::addr0PortName = "address0";
string MemElem::ce1PortName = "ce1";
string MemElem::we1PortName = "we1";
string MemElem::dIn1PortName = "mem_din1";
string MemElem::dOut1PortName = "mem_dout1";
string MemElem::addr1PortName = "address1";
string MemElem::donePortName = "done";
string MemElem::inFileParamName = "TV_IN";
string MemElem::outFileParamName = "TV_OUT";
string MemElem::dataWidthParamName = "DATA_WIDTH";
string MemElem::addrWidthParamName = "ADDR_WIDTH";
string MemElem::dataDepthParamName = "DEPTH";

// class HlsVhdTb

HlsVhdlTb::HlsVhdlTb(const VerificationContext &ctx) : ctx(ctx) {
  duvName = ctx.getVhdlDuvEntityName();
  tleName = ctx.getVhdlDuvEntityName() + "_tb";
  cDuvParams = ctx.getFuvParams();

  Constant hcp("HALF_CLK_PERIOD", "TIME", "2.00 ns");
  constants.push_back(hcp);

  int transNum = getTransactionNumberFromInput();
  logInf(LOG_TAG, "Transaction number computed : " + to_string(transNum));
  if (transNum <= 0) {
    logErr(LOG_TAG, "Invalid number of transactions detected!");
  }
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

      string addrwidthvalue;
      if (ctx.useAddrWidth32) {
        addrwidthvalue = "32";
      } else {
        addrwidthvalue = to_string(((int)ceil(log2(p.arrayLength))));
      }

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
    mElem.we0SignalName = p.parameterName + "_mem_" + mElem.we0PortName;
    mElem.ce0SignalName = p.parameterName + "_mem_" + mElem.ce0PortName;
    mElem.addr0SignalName = p.parameterName + "_mem_" + mElem.addr0PortName;
    mElem.dIn1SignalName = p.parameterName + "_mem_din1";
    mElem.dOut1SignalName = p.parameterName + "_mem_dout1";
    mElem.we1SignalName = p.parameterName + "_mem_" + mElem.we1PortName;
    mElem.ce1SignalName = p.parameterName + "_mem_" + mElem.ce1PortName;
    mElem.addr1SignalName = p.parameterName + "_mem_" + mElem.addr1PortName;
    mElem.doneSignalName = "tb_temp_idle";

    memElems.push_back(mElem);
  }
}

string HlsVhdlTb::getInputFilepathForParam(const CFunctionParameter &param) {
  if (param.isInput) {
    return ctx.getInputVectorPath(param);
  }
  return "";
}

string HlsVhdlTb::getOutputFilepathForParam(const CFunctionParameter &param) {
  if (param.isOutput) {
    return ctx.getVhdlOutPath(param);
  }
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

string HlsVhdlTb::getLibraryHeader() { return vhdlLibraryHeader; }

string HlsVhdlTb::getEntitiyDeclaration() {
  return "entity " + tleName + " is\n\nend entity " + tleName + ";\n";
}

// function to get the port name in the entitiy for each paramter

string HlsVhdlTb::getArchitectureBegin() {
  stringstream arch;
  arch << "architecture behav of " << tleName << " is" << endl << endl;

  arch << "\t-- Constant declarations" << endl << endl;
  arch << getConstantDeclaration() << endl;

  arch << "\t-- Signal declarations" << endl << endl;
  arch << getSignalDeclaration() << endl;

  arch << "begin" << endl << endl;
  return arch.str();
}

string HlsVhdlTb::getConstantDeclaration() {
  stringstream code;
  for (const auto &c : constants) {
    code << "\t"
         << "constant " << c.constName << " : " << c.constType
         << " := " << c.constValue << ";" << endl;
  }
  return code.str();
}

string HlsVhdlTb::getSignalDeclaration() {
  stringstream code;

  code << "\tsignal tb_clk : std_logic := '0';" << endl;
  code << "\tsignal tb_rst : std_logic := '0';" << endl;

  code << "\tsignal tb_start_valid : std_logic := '0';" << endl;
  code << "\tsignal tb_start_ready, tb_started : std_logic;" << endl;

  code << "\tsignal tb_end_valid : std_logic;" << endl;

  code << endl;

  for (size_t i = 0; i < cDuvParams.size(); i++) {
    CFunctionParameter p = cDuvParams[i];
    MemElem m = memElems[i];

    if (m.isArray) {
      code << "\tsignal " << m.ce0SignalName << " : std_logic;" << endl;
      code << "\tsignal " << m.we0SignalName << " : std_logic;" << endl;
      code << "\tsignal " << m.dIn0SignalName << " : std_logic_vector("
           << m.dataWidthParamValue << " - 1 downto 0);" << endl;
      code << "\tsignal " << m.dOut0SignalName << " : std_logic_vector("
           << m.dataWidthParamValue << " - 1 downto 0);" << endl;
      code << "\tsignal " << m.addr0SignalName << " : std_logic_vector("
           << m.addrWidthParamValue << " - 1 downto 0);" << endl
           << endl;

      code << "\tsignal " << m.ce1SignalName << " : std_logic;" << endl;
      code << "\tsignal " << m.we1SignalName << " : std_logic;" << endl;
      code << "\tsignal " << m.dIn1SignalName << " : std_logic_vector("
           << m.dataWidthParamValue << " - 1 downto 0);" << endl;
      code << "\tsignal " << m.dOut1SignalName << " : std_logic_vector("
           << m.dataWidthParamValue << " - 1 downto 0);" << endl;
      code << "\tsignal " << m.addr1SignalName << " : std_logic_vector("
           << m.addrWidthParamValue << " - 1 downto 0);" << endl
           << endl;
    } else {
      if ((cDuvParams[i].isReturn && cDuvParams[i].isOutput) ||
          !cDuvParams[i].isReturn) {
        code << "\tsignal " << m.ce0SignalName << " : std_logic;" << endl;
        code << "\tsignal " << m.we0SignalName << " : std_logic;" << endl;

        code << "\tsignal " << m.dOut0SignalName << " : std_logic_vector("
             << m.dataWidthParamValue << " - 1 downto 0);" << endl;
        code << "\tsignal " << m.dIn0SignalName << " : std_logic_vector("
             << m.dataWidthParamValue << " - 1 downto 0);" << endl;
        code << "\tsignal " << m.dOut0SignalName << "_valid : std_logic;"
             << endl;
        code << "\tsignal " << m.dOut0SignalName << "_ready : std_logic;"
             << endl
             << endl;
      }
    }
  }

  code << endl;

  code << "\tsignal tb_temp_idle : std_logic:= '1';" << endl;
  code << "\tshared variable transaction_idx : INTEGER := 0;" << endl;

  return code.str();
}

string HlsVhdlTb::getMemoryInstanceGeneration() {
  stringstream code;
  for (size_t i = 0; i < memElems.size(); i++) {
    MemElem m = memElems[i];
    CFunctionParameter p = cDuvParams[i];
    if (m.isArray) {
      code << "mem_inst_" << p.parameterName << ":\t entity work.two_port_RAM "
           << endl;
      code << "\tgeneric map(" << endl;
      code << "\t\t" << MemElem::inFileParamName << " => " << m.inFileParamValue
           << "," << endl;
      code << "\t\t" << MemElem::outFileParamName << " => "
           << m.outFileParamValue << "," << endl;
      code << "\t\t" << MemElem::dataDepthParamName << " => "
           << m.dataDepthParamValue << "," << endl;
      code << "\t\t" << MemElem::dataWidthParamName << " => "
           << m.dataWidthParamValue << "," << endl;
      code << "\t\t" << MemElem::addrWidthParamName << " => "
           << m.addrWidthParamValue << endl;
      code << "\t)" << endl;
      code << "\tport map(" << endl;
      code << "\t\t" << MemElem::clkPortName << " => "
           << "tb_clk," << endl;
      code << "\t\t" << MemElem::rstPortName << " => "
           << "tb_rst," << endl;
      code << "\t\t" << MemElem::ce0PortName << " => " << m.ce0SignalName << ","
           << endl;
      code << "\t\t" << MemElem::we0PortName << " => " << m.we0SignalName << ","
           << endl;
      code << "\t\t" << MemElem::addr0PortName << " => " << m.addr0SignalName
           << "," << endl;
      code << "\t\t" << MemElem::dOut0PortName << " => " << m.dOut0SignalName
           << "," << endl;
      code << "\t\t" << MemElem::dIn0PortName << " => " << m.dIn0SignalName
           << "," << endl;
      code << "\t\t" << MemElem::ce1PortName << " => " << m.ce1SignalName << ","
           << endl;
      code << "\t\t" << MemElem::we1PortName << " => " << m.we1SignalName << ","
           << endl;
      code << "\t\t" << MemElem::addr1PortName << " => " << m.addr1SignalName
           << "," << endl;
      code << "\t\t" << MemElem::dOut1PortName << " => " << m.dOut1SignalName
           << "," << endl;
      code << "\t\t" << MemElem::dIn1PortName << " => " << m.dIn1SignalName
           << "," << endl;
      code << "\t\t" << MemElem::donePortName << " => "
           << "tb_temp_idle" << endl;
      code << "\t);" << endl << endl;
    } else {

      if (p.isInput && !p.isOutput && !p.isReturn) {

        code << "arg_inst_" << p.parameterName
             << ":\t entity work.single_argument" << endl;
        code << "\tgeneric map(" << endl;
        code << "\t\t" << MemElem::inFileParamName << " => "
             << m.inFileParamValue << "," << endl;
        code << "\t\t" << MemElem::outFileParamName << " => "
             << m.outFileParamValue << "," << endl;
        code << "\t\t" << MemElem::dataWidthParamName << " => "
             << m.dataWidthParamValue << endl;
        code << "\t)" << endl;
        code << "\tport map(" << endl;
        code << "\t\t" << MemElem::clkPortName << " => "
             << "tb_clk," << endl;
        code << "\t\t" << MemElem::rstPortName << " => "
             << "tb_rst," << endl;
        code << "\t\t" << MemElem::ce0PortName << " => "
             << "'1'"
             << "," << endl;
        code << "\t\t" << MemElem::we0PortName << " => "
             << "'0'"
             << "," << endl;
        code << "\t\t" << MemElem::dOut0PortName << " => " << m.dOut0SignalName
             << "," << endl;
        code << "\t\t" << MemElem::dOut0PortName + "_valid => "
             << m.dOut0SignalName << "_valid," << endl;
        code << "\t\t" << MemElem::dOut0PortName + "_ready => "
             << m.dOut0SignalName << "_ready," << endl;
        code << "\t\t" << MemElem::dIn0PortName << " => "
             << "(others => '0')"
             << "," << endl;
        code << "\t\t" << MemElem::donePortName << " => "
             << "tb_temp_idle" << endl;
        code << "\t);" << endl << endl;
      }
      if (p.isInput && p.isOutput && !p.isReturn) {

        code << "arg_inst_" << p.parameterName
             << ":\t entity work.single_argument" << endl;
        code << "\tgeneric map(" << endl;
        code << "\t\t" << MemElem::inFileParamName << " => "
             << m.inFileParamValue << "," << endl;
        code << "\t\t" << MemElem::outFileParamName << " => "
             << m.outFileParamValue << "," << endl;
        code << "\t\t" << MemElem::dataWidthParamName << " => "
             << m.dataWidthParamValue << endl;
        code << "\t)" << endl;
        code << "\tport map(" << endl;
        code << "\t\t" << MemElem::clkPortName << " => "
             << "tb_clk," << endl;
        code << "\t\t" << MemElem::rstPortName << " => "
             << "tb_rst," << endl;
        code << "\t\t" << MemElem::ce0PortName << " => "
             << "'1'"
             << "," << endl;
        code << "\t\t" << MemElem::we0PortName << " => " << m.we0SignalName
             << "," << endl;
        code << "\t\t" << MemElem::dOut0PortName << " => " << m.dOut0SignalName
             << "," << endl;
        code << "\t\t" << MemElem::dOut0PortName + "_valid => "
             << m.dOut0SignalName << "_valid," << endl;
        code << "\t\t" << MemElem::dOut0PortName + "_ready => "
             << m.dOut0SignalName << "_ready," << endl;
        code << "\t\t" << MemElem::dIn0PortName << " => " << m.dIn0SignalName
             << "," << endl;
        code << "\t\t" << MemElem::donePortName << " => "
             << "tb_temp_idle" << endl;
        code << "\t);" << endl << endl;
      }

      if (!p.isInput && p.isOutput && p.isReturn) {

        code << "arg_inst_" << p.parameterName
             << ":\t entity work.single_argument" << endl;
        code << "\tgeneric map(" << endl;
        code << "\t\t" << MemElem::inFileParamName << " => "
             << m.inFileParamValue << "," << endl;
        code << "\t\t" << MemElem::outFileParamName << " => "
             << m.outFileParamValue << "," << endl;
        code << "\t\t" << MemElem::dataWidthParamName << " => "
             << m.dataWidthParamValue << endl;
        code << "\t)" << endl;
        code << "\tport map(" << endl;
        code << "\t\t" << MemElem::clkPortName << " => "
             << "tb_clk," << endl;
        code << "\t\t" << MemElem::rstPortName << " => "
             << "tb_rst," << endl;
        code << "\t\t" << MemElem::ce0PortName << " => "
             << "'1'"
             << "," << endl;
        code << "\t\t" << MemElem::we0PortName << " => "
             << "tb_end_valid"
             << "," << endl;
        code << "\t\t" << MemElem::dOut0PortName << " => " << m.dOut0SignalName
             << "," << endl;
        code << "\t\t" << MemElem::dOut0PortName + "_valid => "
             << m.dOut0SignalName << "_valid," << endl;
        code << "\t\t" << MemElem::dOut0PortName + "_ready => "
             << m.dOut0SignalName << "_ready," << endl;
        code << "\t\t" << MemElem::dIn0PortName << " => " << m.dIn0SignalName
             << "," << endl;
        code << "\t\t" << MemElem::donePortName << " => "
             << "tb_temp_idle" << endl;
        code << "\t);" << endl << endl;
      }

      if (!p.isInput && p.isOutput && !p.isReturn) {

        code << "arg_inst_" << p.parameterName
             << ":\t entity work.single_argument" << endl;
        code << "\tgeneric map(" << endl;
        code << "\t\t" << MemElem::inFileParamName << " => "
             << m.inFileParamValue << "," << endl;
        code << "\t\t" << MemElem::outFileParamName << " => "
             << m.outFileParamValue << "," << endl;
        code << "\t\t" << MemElem::dataWidthParamName << " => "
             << m.dataWidthParamValue << endl;
        code << "\t)" << endl;
        code << "\tport map(" << endl;
        code << "\t\t" << MemElem::clkPortName << " => "
             << "tb_clk," << endl;
        code << "\t\t" << MemElem::rstPortName << " => "
             << "tb_rst," << endl;
        code << "\t\t" << MemElem::ce0PortName << " => "
             << "'1'"
             << "," << endl;
        code << "\t\t" << MemElem::we0PortName << " => " << m.we0SignalName
             << "," << endl;
        code << "\t\t" << MemElem::dIn0PortName << " => " << m.dIn0SignalName
             << "," << endl;
        code << "\t\t" << MemElem::dOut0PortName << " => " << m.dOut0SignalName
             << "," << endl;
        code << "\t\t" << MemElem::dOut0PortName + "_valid => "
             << m.dOut0SignalName << "_valid," << endl;
        code << "\t\t" << MemElem::dOut0PortName + "_ready => "
             << m.dOut0SignalName << "_ready," << endl;
        code << "\t\t" << MemElem::donePortName << " => "
             << "tb_temp_idle" << endl;
        code << "\t);" << endl << endl;
      }
    }
  }
  return code.str();
}

string HlsVhdlTb::getMemComponentDeclration() {
  stringstream code;
  code << "\tcomponent two_port_RAM is" << endl;
  code << "\tgeneric(" << endl;
  code << "\t\t" << MemElem::inFileParamName << " : STRING;" << endl;
  code << "\t\t" << MemElem::outFileParamName << " : STRING;" << endl;
  code << "\t\t" << MemElem::dataWidthParamName << " : INTEGER;" << endl;
  code << "\t\t" << MemElem::addrWidthParamName << " : INTEGER;" << endl;
  code << "\t\t" << MemElem::dataDepthParamName << " : INTEGER" << endl;
  code << "\t);" << endl;
  code << "\tport(" << endl;
  code << "\t\t" << MemElem::clkPortName << " : in std_logic;" << endl;
  code << "\t\t" << MemElem::rstPortName << " : in std_logic;" << endl;
  code << "\t\t" << MemElem::ce0PortName << " : in std_logic;" << endl;
  code << "\t\t" << MemElem::we0PortName << " : in std_logic;" << endl;
  code << "\t\t" << MemElem::addr0PortName << " : in std_logic_vector("
       << MemElem::addrWidthParamName << " - 1 downto 0);" << endl;
  code << "\t\t" << MemElem::dOut0PortName << " : out std_logic_vector("
       << MemElem::dataWidthParamName << " - 1 downto 0);" << endl;
  code << "\t\t" << MemElem::dIn0PortName << " : in std_logic_vector("
       << MemElem::dataWidthParamName << " - 1 downto 0);" << endl;
  code << "\t\t" << MemElem::ce1PortName << " : in std_logic;" << endl;
  code << "\t\t" << MemElem::we1PortName << " : in std_logic;" << endl;
  code << "\t\t" << MemElem::addr1PortName << " : in std_logic_vector("
       << MemElem::addrWidthParamName << " - 1 downto 0);" << endl;
  code << "\t\t" << MemElem::dOut1PortName << " : out std_logic_vector("
       << MemElem::dataWidthParamName << " - 1 downto 0);" << endl;
  code << "\t\t" << MemElem::dIn1PortName << " : in std_logic_vector("
       << MemElem::dataWidthParamName << " - 1 downto 0);" << endl;
  code << "\t\t" << MemElem::donePortName << " : in std_logic" << endl;
  code << "\t);" << endl;
  code << "\tend component two_port_RAM;" << endl;
  return code.str();
}

string HlsVhdlTb::getArgComponentDeclration() {
  stringstream code;
  code << "\tcomponent single_argument is" << endl;
  code << "\tgeneric(" << endl;
  code << "\t\t" << MemElem::inFileParamName << " : STRING;" << endl;
  code << "\t\t" << MemElem::outFileParamName << " : STRING;" << endl;
  code << "\t\t" << MemElem::dataWidthParamName << " : INTEGER" << endl;
  code << "\t);" << endl;
  code << "\tport(" << endl;
  code << "\t\t" << MemElem::clkPortName << " : in std_logic;" << endl;
  code << "\t\t" << MemElem::rstPortName << " : in std_logic;" << endl;
  code << "\t\t" << MemElem::ce0PortName << " : in std_logic;" << endl;
  code << "\t\t" << MemElem::we0PortName << " : in std_logic;" << endl;
  code << "\t\t" << MemElem::dIn0PortName << " : in std_logic_vector("
       << MemElem::dataWidthParamName << " - 1 downto 0);" << endl;
  code << "\t\t" << MemElem::dOut0PortName << " : out std_logic_vector("
       << MemElem::dataWidthParamName << " - 1 downto 0);" << endl;
  code << "\t\t" << MemElem::donePortName << " : in std_logic" << endl;
  code << "\t);" << endl;
  code << "\tend component single_argument;" << endl;
  return code.str();
}

string HlsVhdlTb::getDuvInstanceGeneration() {
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
    } else {
      if (p.isInput) {
        if (ctx.experimental) {
          duvPortMap.emplace_back(p.parameterName, m.dOut0SignalName);
          duvPortMap.emplace_back(p.parameterName + "_valid",
                                  m.dOut0SignalName + "_valid");
          duvPortMap.emplace_back(p.parameterName + "_ready",
                                  m.dOut0SignalName + "_ready");
        } else {
          duvPortMap.emplace_back(getValidInPortNameForCParam(p.parameterName),
                                  "'1'");
          duvPortMap.emplace_back(getDataInSaPortNameForCParam(p.parameterName),
                                  m.dOut0SignalName);
        }
      }

      if (p.isOutput) {
        if (p.isReturn) {
          if (ctx.experimental) {
            duvPortMap.emplace_back("out0", m.dIn0SignalName);
            duvPortMap.emplace_back("out0_valid", "tb_end_valid");
            duvPortMap.emplace_back("out0_ready", "'1'");
          } else {
            duvPortMap.emplace_back("end_out", m.dIn0SignalName);
            duvPortMap.emplace_back("end_valid", "tb_end_valid");
            duvPortMap.emplace_back("end_ready", "'1'");
          }
        } else {
          duvPortMap.emplace_back(getValidOutPortNameForCParam(p.parameterName),
                                  m.we0SignalName);
          duvPortMap.emplace_back(
              getDataOutSaPortNameForCParam(p.parameterName), m.dIn0SignalName);
          duvPortMap.emplace_back(getReadyInPortNameForCParam(p.parameterName),
                                  "'1'");
        }
      } else if (p.isReturn) {
        duvPortMap.emplace_back("end_valid", "tb_end_valid");
        duvPortMap.emplace_back("end_ready", "'1'");
      }
    }
  }

  // Start signal
  if (!ctx.experimental)
    duvPortMap.emplace_back("start_in", "(others => '0')");
  duvPortMap.emplace_back("start_ready", "tb_start_ready");
  duvPortMap.emplace_back("start_valid", "tb_start_valid");

  stringstream code;
  code << "duv: \t entity work." << duvName << endl;
  code << "\t\tport map (" << endl;
  for (size_t i = 0; i < duvPortMap.size(); i++) {
    pair<string, string> elem = duvPortMap[i];
    code << "\t\t\t" << elem.first << " => " << elem.second
         << ((i < duvPortMap.size() - 1) ? "," : "") << endl;
  }
  code << "\t\t);" << endl << endl;
  return code.str();
}

string HlsVhdlTb::getDuvComponentDeclaration() {
  stringstream code;
  code << "\tcomponent " << duvName << " is" << endl;
  code << "\tport(" << endl;
  code << "\t\tclk : in std_logic;" << endl;
  code << "\t\trst : in std_logic;" << endl;

  for (size_t i = 0; i < cDuvParams.size(); i++) {
    CFunctionParameter p = cDuvParams[i];
    MemElem m = memElems[i];

    if (m.isArray) {
      code << "\t\t" << getAddr0PortNameForCParam(p.parameterName)
           << " : out std_logic_vector(" << ((int)ceil(log2(p.arrayLength)) - 1)
           << " downto 0);" << endl;
      code << "\t\t" << getCe0PortNameForCParam(p.parameterName)
           << " : out std_logic;" << endl;
      code << "\t\t" << getWe0PortNameForCParam(p.parameterName)
           << " : out std_logic;" << endl;
      code << "\t\t" << getDataOut0PortNameForCParam(p.parameterName)
           << " : out std_logic_vector(" << ((int)p.dtWidth - 1)
           << " downto 0);" << endl;
      code << "\t\t" << getDataIn0PortNameForCParam(p.parameterName)
           << " : in std_logic_vector(" << ((int)p.dtWidth - 1) << " downto 0);"
           << endl;

      code << "\t\t" << getAddr1PortNameForCParam(p.parameterName)
           << " : out std_logic_vector(" << ((int)ceil(log2(p.arrayLength)) - 1)
           << " downto 0);" << endl;
      code << "\t\t" << getCe1PortNameForCParam(p.parameterName)
           << " : out std_logic;" << endl;
      code << "\t\t" << getWe1PortNameForCParam(p.parameterName)
           << " : out std_logic;" << endl;
      code << "\t\t" << getDataOut1PortNameForCParam(p.parameterName)
           << " : out std_logic_vector(" << ((int)p.dtWidth - 1)
           << " downto 0);" << endl;
      code << "\t\t" << getDataIn1PortNameForCParam(p.parameterName)
           << " : in std_logic_vector(" << ((int)p.dtWidth - 1) << " downto 0);"
           << endl;

    } else {

      if (!m.isArray && p.isInput) {
        code << "\t\t" << getValidInPortNameForCParam(p.parameterName)
             << " : in std_logic;" << endl;
        code << "\t\t" << getDataIn0PortNameForCParam(p.parameterName)
             << " : in std_logic_vector(" << ((int)p.dtWidth - 1)
             << " downto 0);" << endl;
      }

      if (!m.isArray && p.isOutput && p.isReturn) {
        code << "\t\t"
             << "end_out : out std_logic_vector(" << ((int)p.dtWidth - 1)
             << " downto 0);" << endl;
        code << "\t\t"
             << "end_valid : out std_logic;" << endl;
        code << "\t\t"
             << "end_ready : in std_logic;" << endl;
      }

      if (!m.isArray && !p.isOutput && p.isReturn) {
        code << "\t\t"
             << "end_valid : out std_logic;" << endl;
        code << "\t\t"
             << "end_ready : in std_logic;" << endl;
      }

      if (!m.isArray && p.isOutput && !p.isReturn) {
        code << "\t\t" << getValidOutPortNameForCParam(p.parameterName)
             << " : out std_logic;" << endl;
        code << "\t\t" << getDataOut0PortNameForCParam(p.parameterName)
             << " : out std_logic_vector(" << ((int)p.dtWidth - 1)
             << " downto 0);" << endl;
        code << "\t\t" << getReadyInPortNameForCParam(p.parameterName)
             << " : in std_logic;" << endl;
      }
    }
  }
  code << "\t\tstart_valid : in std_logic;" << endl;
  code << "\t\tstart_ready : out std_logic" << endl;
  code << "\t);" << endl;
  code << "\tend component " << duvName << ";" << endl;
  return code.str();
}

string HlsVhdlTb::getCommonBody() { return commonTbBody; }

string HlsVhdlTb::getArchitectureEnd() { return "end architecture behav;\n"; }

string HlsVhdlTb::getOutputTagGeneration() {
  stringstream out;
  out << "\n";
  out << "---------------------------------------------------------------------"
         "-------\n";
  out << "-- Write \"[[[runtime]]]\" and \"[[[/runtime]]]\" for output "
         "transactor\n";
  for (auto &cDuvParam : cDuvParams) {
    if (cDuvParam.isOutput) {
      out << "write_output_transactor_" << cDuvParam.parameterName
          << "_runtime_proc : process\n";
      out << "	file fp             : TEXT;\n";
      out << "	variable fstatus    : FILE_OPEN_STATUS;\n";
      out << "	variable token_line : LINE;\n";
      out << "	variable token      : STRING(1 to 1024);\n";
      out << "\n";
      out << "begin\n";
      out << "	file_open(fstatus, fp, OUTPUT_" << cDuvParam.parameterName
          << ", WRITE_MODE);\n";
      out << "	if (fstatus /= OPEN_OK) then\n";
      out << "		assert false report \"Open file \" & OUTPUT_"
          << cDuvParam.parameterName << " & \" failed!!!\" severity note;\n";
      out << "		assert false report \"ERROR: Simulation using HLS TB "
             "failed.\" severity failure;\n";
      out << "	end if;\n";
      out << "	write(token_line, string'(\"[[[runtime]]]\"));\n";
      out << "	writeline(fp, token_line);\n";
      out << "	file_close(fp);\n";
      out << "	while transaction_idx /= TRANSACTION_NUM loop\n";
      out << "		wait until tb_clk'event and tb_clk = '1';\n";
      out << "	end loop;\n";
      out << "	wait until tb_clk'event and tb_clk = '1';\n";
      out << "	wait until tb_clk'event and tb_clk = '1';\n";
      out << "	file_open(fstatus, fp, OUTPUT_" << cDuvParam.parameterName
          << ", APPEND_MODE);\n";
      out << "	if (fstatus /= OPEN_OK) then\n";
      out << "		assert false report \"Open file \" & OUTPUT_"
          << cDuvParam.parameterName << " & \" failed!!!\" severity note;\n";
      out << "		assert false report \"ERROR: Simulation using HLS TB "
             "failed.\" severity failure;\n";
      out << "	end if;\n";
      out << "	write(token_line, string'(\"[[[/runtime]]]\"));\n";
      out << "	writeline(fp, token_line);\n";
      out << "	file_close(fp);\n";
      out << "	wait;\n";
      out << "end process;\n";
    }
  }
  out << "---------------------------------------------------------------------"
         "-------\n\n";
  return out.str();
}

string HlsVhdlTb::generateVhdlTestbench() {
  stringstream tbOut;
  tbOut << getLibraryHeader() << endl;
  tbOut << getEntitiyDeclaration() << endl;
  tbOut << getArchitectureBegin() << endl;
  tbOut << getDuvInstanceGeneration() << endl;
  tbOut << getMemoryInstanceGeneration() << endl;
  tbOut << getOutputTagGeneration() << endl;
  tbOut << getCommonBody() << endl;
  tbOut << getArchitectureEnd() << endl;
  return tbOut.str();
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
