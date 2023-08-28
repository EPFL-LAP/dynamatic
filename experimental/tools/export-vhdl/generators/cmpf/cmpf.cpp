//===- cmpf.cpp - Fulfill a cmpf generator ----------------------*- C++ -*-===//
//
// Experimental tool that realizes a generator for float comparison component
// (cmpf)
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/EpochTracker.h"
#include "llvm/Support/AlignOf.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemAlloc.h"
#include "llvm/Support/ReverseIteration.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/type_traits.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace llvm;

int main(int argc, char **argv) {
  // no predicate provided
  if (argc < 2) {
    llvm::errs() << "Too few arguments in generator call\n";
    return 1;
  }
  // map which stores corresponding data for predicates
  StringMap<std::string> data = {
      {"oeq", "00001"}, {"ogt", "00010"}, {"oge", "00011"}, {"olt", "00100"},
      {"ole", "00101"}, {"one", "00110"}, {"ord", "00111"}, {"ueq", "01000"},
      {"ugt", "01001"}, {"uge", "01010"}, {"ult", "01011"}, {"ule", "01100"},
      {"une", "01101"}, {"uno", "01110"}};

  // necessary file
  std::string cmpf_file =
      "library IEEE;\nuse IEEE.std_logic_1164.all;\nuse "
      "ieee.numeric_std.all;\nuse work.customTypes.all;\n\nentity "
      "cmpf_#PREDICATE# is\n "
      " generic (\n    BITWIDTH : integer\n  );\n  port (\n    -- inputs\n    "
      "clk          : in std_logic;\n    rst          : in std_logic;\n    lhs "
      "         : in std_logic_vector(BITWIDTH - 1 downto 0);\n    lhs_valid   "
      " : in std_logic;\n    rhs          : in std_logic_vector(BITWIDTH - 1 "
      "downto 0);\n    rhs_valid    : in std_logic;\n    result_ready : in "
      "std_logic;\n    -- outputs\n    lhs_ready    : out std_logic;\n    "
      "rhs_ready    : out std_logic;\n    result       : out "
      "std_logic_vector(BITWIDTH - 1 downto 0);\n    result_valid : out "
      "std_logic);\nend entity;\n\narchitecture arch of cmpf_#PREDICATE# "
      "is\n\n  component array_RAM_fcmp_32cud is\n    generic (\n      ID      "
      "   : integer := 1;\n      NUM_STAGE  : integer := 2;\n      din0_WIDTH "
      ": integer := 32;\n      din1_WIDTH : integer := 32;\n      dout_WIDTH : "
      "integer := 1\n    );\n    port (\n      clk    : in std_logic;\n      "
      "reset  : in std_logic;\n      ce     : in std_logic;\n      din0   : in "
      "std_logic_vector(din0_WIDTH - 1 downto 0);\n      din1   : in "
      "std_logic_vector(din1_WIDTH - 1 downto 0);\n      opcode : in "
      "std_logic_vector(4 downto 0);\n      dout   : out "
      "std_logic_vector(dout_WIDTH - 1 downto 0)\n    );\n  end component;\n\n "
      " signal join_valid   : std_logic;\n  constant alu_opcode : "
      "std_logic_vector(4 downto 0) := \"#CONST#\";\n\nbegin\n\n  "
      "result(BITWIDTH - 1 downto 1) <= (others => '0');\n\n  "
      "array_RAM_fcmp_32ns_32ns_1_2_1_u1 : component array_RAM_fcmp_32cud\n    "
      "generic map(\n      ID         => 1,\n      NUM_STAGE  => 2,\n      "
      "din0_WIDTH => 32,\n      din1_WIDTH => 32,\n      dout_WIDTH => 1)\n    "
      "port map(\n      clk     => clk,\n      reset   => rst,\n      din0    "
      "=> lhs,\n      din1    => rhs,\n      ce      => result_ready,\n      "
      "opcode  => alu_opcode,\n      dout(0) => result(0));\n\n    "
      "join_write_temp : entity work.join(arch) generic map(2)\n      port "
      "map(\n      (lhs_valid,\n        rhs_valid),\n        result_ready,\n   "
      "     join_valid,\n        (lhs_ready,\n        rhs_ready));\n\n    buff "
      ": entity work.delay_buffer(arch)\n      generic map(1)\n      port "
      "map(\n        clk,\n        rst,\n        join_valid,\n        "
      "result_ready,\n        result_valid);\n  end architecture;\n";

  // get the predicate name from command line options
  std::string predicateName(argv[1]);
  std::stringstream buffer;
  std::string modText;
  auto it = data.find(predicateName);
  if (it == data.end()) {
    // wrong predicate
    llvm::errs() << "Wrong predicate\n";
    return 1;
  } else {
    // oeq, ogt, oge, olt, ole, one, ord, ueq, ugt, uge, ult, ule, une, uno
    buffer << cmpf_file;
    std::string temp{};
    // replace all #---# sequences
    while (buffer.good()) {
      std::getline(buffer, temp, '#');
      modText += temp;
      if (!buffer.good())
        break;

      std::getline(buffer, temp, '#');
      if (temp == "PREDICATE")
        modText += it->getKey().str();
      else
        modText += it->getValue();
    }
  }
  // print the result module text to std output
  llvm::outs() << modText;
  return 0;
}
