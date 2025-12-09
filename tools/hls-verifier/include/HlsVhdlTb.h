//===- HlsVhdlTb.h ----------------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HLS_VERIFIER_HLS_VHDL_TB_H
#define HLS_VERIFIER_HLS_VHDL_TB_H

#include "VerificationContext.h"
#include "dynamatic/Support/Utils/Utils.h"
#include "mlir/Support/IndentedOstream.h"
#include <boost/dynamic_bitset.hpp>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

using namespace std;
static const string VHDL_LIBRARY_HEADER = R"DELIM(
library IEEE;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_textio.all;
use ieee.numeric_std.all;
use std.textio.all;
use work.sim_package.all;
entity tb is
end entity tb;

architecture behavior of tb is

)DELIM";

static const string VERILOG_LIBRARY_HEADER = R"DELIM(
`timescale 1ns/1ps
module tb;

)DELIM";

static const string VHDL_COMMON_TB_BODY = R"DELIM(

generate_sim_done_proc : process
begin
  while (transaction_idx /= TRANSACTION_NUM) loop
    wait until tb_clk'event and tb_clk = '1';
  end loop;
  wait until tb_clk'event and tb_clk = '1';
  wait until tb_clk'event and tb_clk = '1';
  wait until tb_clk'event and tb_clk = '1';
  assert false
  report "Simulation done! Latency = " & integer'image((now - RESET_LATENCY) / (2 * HALF_CLK_PERIOD)) & " cycles"
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
  wait for RESET_LATENCY;
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
    if (tb_start_valid = '1') then
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

static const string VERILOG_COMMON_TB_BODY = R"DELIM(
// Equivalent of VHDL generate_sim_done_proc
initial begin
  while (transaction_idx != TRANSACTION_NUM) begin
      @(posedge tb_clk);
  end

  repeat(3)@(posedge tb_clk);

  $display("Simulation done! Latency = %0d cycles",
           ($time - RESET_LATENCY) / (2 * HALF_CLK_PERIOD));

  $display("NORMAL EXIT (note: failure is to force the simulator to stop)");
  $finish;

  forever @(posedge tb_clk);
end

// Equivalent of VHDL gen_clock_proc
initial begin
    tb_clk = 1'b0;
    forever begin
        #HALF_CLK_PERIOD tb_clk = ~tb_clk;
    end
end

// Equivalent of VHDL gen_reset_proc
initial begin
    tb_rst = 1'b1;
    #RESET_LATENCY;
    tb_rst = 1'b0;
end

// Equivalent of VHDL acknowledge_tb_end
always @(posedge tb_clk or posedge tb_rst) begin
    if (tb_rst) begin
        tb_global_ready <= 1'b1;
        tb_stop <= 1'b0;
    end else begin
        if (tb_global_valid) begin
            tb_global_ready <= 1'b0;
            tb_stop <= 1'b1;
        end
    end
end

// Equivalent of VHDL generate_idle_signal
always @(posedge tb_clk or posedge tb_rst) begin
    if (tb_rst) begin
        tb_temp_idle <= 1'b1;
    end else begin
        // Default keep
        tb_temp_idle <= tb_temp_idle;

        if (tb_start_valid)
            tb_temp_idle <= 1'b0;

        if (tb_stop)
            tb_temp_idle <= 1'b1;
    end
end

// Equivalent of VHDL generate_start_signal
always @(posedge tb_clk or posedge tb_rst) begin
    if (tb_rst) begin
        tb_start_valid <= 1'b0;
        tb_started <= 1'b0;
    end else begin
        if (!tb_started) begin
            tb_start_valid <= 1'b1;
            tb_started <= 1'b1;
        end else begin
            tb_start_valid <= tb_start_valid & (~tb_start_ready);
        end
    end
end

// Equivalent of transaction_increment
initial begin
    wait (tb_rst == 1'b0);

    while (tb_temp_idle != 1'b1)
        @(posedge tb_clk);

    // Wait until first non-idle
    wait (tb_temp_idle == 1'b0);

    forever begin
        while (tb_temp_idle != 1'b1)
            @(posedge tb_clk);

        transaction_idx = transaction_idx + 1;

        wait (tb_temp_idle == 1'b0);
    end
end

)DELIM";

static const string VHDL_PROC_WRITE_TRANSACTIONS = R"DELIM(
write_output_transactor_{0}_runtime_proc : process
  file fp             : TEXT;
  variable fstatus    : FILE_OPEN_STATUS;
  variable token_line : LINE;
  variable token      : STRING(1 to 1024);
begin
  file_open(fstatus, fp, OUTPUT_{0} , WRITE_MODE);
  if (fstatus /= OPEN_OK) then
    assert false report "Open file " & OUTPUT_{0} & " failed!!!" severity note;
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
  file_open(fstatus, fp, OUTPUT_{0}, APPEND_MODE);
  if (fstatus /= OPEN_OK) then
    assert false report "Open file " & OUTPUT_{0} & " failed!!!" severity note;
    assert false report "ERROR: Simulation using HLS TB failed." severity failure;
  end if;
  write(token_line, string'("[[[/runtime]]]"));
  writeline(fp, token_line);
  file_close(fp);
  wait;
end process;
)DELIM";

static const string VERILOG_PROC_WRITE_TRANSACTIONS = R"DELIM(
// Equivalent of VHDL write_output_transactor_{0}_runtime_proc
integer fp_{0};

initial begin
    fp_{0} = $fopen(OUTPUT_{0}, "w");
    if (fp_{0} == 0) begin
        $display("Open file %0s failed!!!", OUTPUT_{0});
        $display("ERROR: Simulation using HLS TB failed.");
        $finish;
    end

    // Write header
    $fdisplay(fp_{0}, "[[[runtime]]]");
    $fclose(fp_{0});

    while (transaction_idx != TRANSACTION_NUM) begin
        @(posedge tb_clk);
    end

    @(posedge tb_clk);
    @(posedge tb_clk);

    fp_{0} = $fopen(OUTPUT_{0}, "a");  // append mode
    if (fp_{0} == 0) begin
        $display("Open file %0s failed!!!", OUTPUT_{0});
        $display("ERROR: Simulation using HLS TB failed.");
        $finish;
    end

    $fdisplay(fp_{0}, "[[[/runtime]]]");
    $fclose(fp_{0});

    forever @(posedge tb_clk);
end
)DELIM";

enum TypeConstant { INTEGER, STRING, TIME };

enum SpecialSignal { CONST_ZERO, CONST_ONE, OPEN };

enum SimulatorSignalType { REGISTER, WIRE };

std::string toBinaryString(int initialValue, unsigned int bitwidth);

// Get input argument of type Ty and their associated names
template <typename Ty>
llvm::SmallVector<std::pair<Ty, std::string>>
getInputArguments(handshake::FuncOp *funcOp) {
  llvm::SmallVector<std::pair<Ty, std::string>> interfaces;
  for (auto [arg, portAttr] : llvm::zip_equal(
           funcOp->getBodyBlock()->getArguments(), funcOp->getArgNames())) {
    if (Ty type = dyn_cast<Ty>(arg.getType())) {
      std::string argName = portAttr.dyn_cast<StringAttr>().data();
      interfaces.emplace_back(type, argName);
    }
  }
  return interfaces;
}

template <typename Ty>
llvm::SmallVector<std::pair<Ty, std::string>>
getOutputArguments(handshake::FuncOp *funcOp) {
  llvm::SmallVector<std::pair<Ty, std::string>> interfaces;

  for (auto [resType, portAttr] :
       llvm::zip_equal(funcOp->getResultTypes(), funcOp->getResNames())) {
    std::string argName = portAttr.dyn_cast<StringAttr>().str();
    if (Ty type = dyn_cast<Ty>(resType)) {
      interfaces.emplace_back(type, argName);
    }
  }
  return interfaces;
}

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

using ConnectedValueType = std::variant<std::string, SpecialSignal>;

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
  std::vector<std::pair<std::string, ConnectedValueType>> connections;
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

  Instance &connect(const std::string &port, SpecialSignal value) {
    connections.emplace_back(port, value);
    return *this;
  }

  void emitVhdl(mlir::raw_indented_ostream &os, VerificationContext &ctx) {

    // VHDL port map needs a continuous assignment
    std::sort(connections.begin(), connections.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    if (ctx.simLanguage == VHDL) {
      os << instanceName << ": entity work." << moduleName << "\n";

      if (!params.empty()) {
        os << "generic map(\n";
        os.indent();
        for (size_t i = 0; i < params.size(); ++i) {
          os << params[i].first << " => " << params[i].second;
          if (i != params.size() - 1)
            os << ",";
          os << "\n";
        }
        os.unindent();
        os << ")\n";
      }
      os << "port map(\n";
      os.indent();
      for (size_t i = 0; i < connections.size(); ++i) {
        const auto &[port, sig] = connections[i];
        os << port << " => ";

        std::visit(
            [&](auto &&v) {
              using T = std::decay_t<decltype(v)>;
              if constexpr (std::is_same_v<T, SpecialSignal>) {
                switch (v) {
                case CONST_ZERO:
                  os << "'0'";
                  break;
                case CONST_ONE:
                  os << "'1'";
                  break;
                case OPEN:
                  os << "open";
                  break;
                }
              } else {
                os << v;
              }
            },
            sig);

        if (i != connections.size() - 1)
          os << ",";
        os << "\n";
      }
    }

    if (ctx.simLanguage == VERILOG) {
      os << moduleName;

      if (!params.empty()) {
        os << " #(\n";
        os.indent();

        for (size_t i = 0; i < params.size(); ++i) {
          os << "." << params[i].first << "(" << params[i].second << ")";
          if (i != params.size() - 1)
            os << ",";
          os << "\n";
        }

        os.unindent();
        os << ")";
      }

      os << " " << instanceName << " (\n";
      os.indent();

      for (size_t i = 0; i < connections.size(); ++i) {
        const auto &[port, sig] = connections[i];
        os << "." << port;

        std::visit(
            [&](auto &&v) {
              using T = std::decay_t<decltype(v)>;
              if constexpr (std::is_same_v<T, SpecialSignal>) {
                switch (v) {
                case CONST_ZERO:
                  os << "(1'b0)";
                  break;
                case CONST_ONE:
                  os << "(1'b1)";
                  break;
                case OPEN:
                  os << "()";
                  break;
                }
              } else {
                os << "(" << v << ")";
              }
            },
            sig);
        if (i != connections.size() - 1)
          os << ",";
        os << "\n";
      }
    }
    os.unindent();
    os << ");\n\n";
  }
};

void vhdlTbCodegen(VerificationContext &ctx);

inline void
verilogSignalDeclaration(mlir::raw_indented_ostream &os, const string &name,
                         std::optional<unsigned int> size = std::nullopt,
                         std::optional<int> initialValue = std::nullopt) {
  if (size)
    os << "[" << *size << "-1 : 0] ";
  os << name;
  if (initialValue) {
    unsigned int bitwidth = size ? *size : 1;
    if (bitwidth > 1) {
      std::string binaryString = toBinaryString(*initialValue, bitwidth);
      os << " = " << bitwidth << "'b" << binaryString;
    } else {
      os << " = " << initialValue;
    }
  }
  os << ";\n";
}

// Declare a signal. Usage:
// - declareSTL(os, "signal_name"); declares a std_logic signal
// - declareSTL(os, "signal_name", "SIGNAL_WIDTH"); declares a std_logic_vector
inline void declareSTL(VerificationContext &ctx, mlir::raw_indented_ostream &os,
                       const string &name,
                       std::optional<unsigned int> size = std::nullopt,
                       std::optional<int> initialValue = std::nullopt) {
  if (ctx.simLanguage == VHDL) {
    os << "signal " << name << " : std_logic";
    if (size)
      os << "_vector(" << *size << " - 1 downto 0)";
    if (initialValue) {
      // initialValue can be of type std::string or int
      // int is formatted with ' '. Example: signal name : std_logic := '0';
      // std::string is formatted without ' '. Example:
      // signal a_din1 : std_logic_vector(32 - 1 downto 0) :=  (others => '0');
      // bitwidth is size, if it exists, else 1

      unsigned int bitwidth = size ? *size : 1;
      if (bitwidth > 1) {
        std::string binaryString = toBinaryString(*initialValue, bitwidth);
        os << " :=  \"" << binaryString << "\"";
      } else {
        os << " := '" << *initialValue << "'";
      }
    }
    os << ";\n";
  }
  if (ctx.simLanguage == VERILOG) {
    os << "wire ";
    verilogSignalDeclaration(os, name, size, initialValue);
    /*
    if (size)
      os << "[" << *size << "-1 : 0] ";
    os << name;
    if (initialValue) {
      unsigned int bitwidth = size ? *size : 1;
      if (bitwidth > 1) {
        std::string binaryString = toBinaryString(*initialValue, bitwidth);
        os << " = " << bitwidth << "'b" << binaryString;
      } else {
        os << " = " << initialValue;
      }
    }
    os << ";\n";
    */
  }
}

inline void declareReg(VerificationContext &ctx, mlir::raw_indented_ostream &os,
                       const string &name,
                       std::optional<unsigned int> size = std::nullopt,
                       std::optional<int> initialValue = std::nullopt) {

  if (ctx.simLanguage == VHDL) {
    declareSTL(ctx, os, name, size, initialValue);
  }
  if (ctx.simLanguage == VERILOG) {
    os << "reg ";
    verilogSignalDeclaration(os, name, size, initialValue);
  }
}

inline void declareConstant(VerificationContext &ctx,
                            mlir::raw_indented_ostream &os, const string &name,
                            TypeConstant type, const string &value) {
  if (ctx.simLanguage == VHDL) {
    os << "constant " << name << " : ";
    switch (type) {
    case INTEGER:
      os << "INTEGER := " << value;
      break;

    case STRING:
      os << "STRING := " << value;
      break;

    case TIME:
      os << "TIME := " << value << " ns";
    }
    os << ";\n";
  }
  if (ctx.simLanguage == VERILOG) {
    os << "parameter " << name << " = " << value << ";\n";
  }
}

#endif // HLS_VERIFIER_HLS_VHDL_TB_H
