//===- cmpi.cpp - Fulfill a cmpi generator ----------------------*- C++ -*-===//
//
// Experimental tool that realizes a generator for integer comparison component
// (cmpi)
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
  // maps which store corresponding data for predicates
  StringMap<std::pair<std::string, std::string>> equality = {
      {"eq", {"one", "zero"}}, {"ne", {"zero", "one"}}};
  StringMap<std::string> comparison = {
      {"sge", ">="}, {"sgt", ">"}, {"sle", "<="}, {"slt", "<"},
      {"uge", ">="}, {"ugt", ">"}, {"ule", "<="}, {"ult", "<"}};

  // necessary files
  std::string equality_file =
      "library IEEE;\nuse IEEE.std_logic_1164.all;\nuse "
      "ieee.numeric_std.all;\nuse work.customTypes.all;\n\nentity "
      "cmpi_#PREDICATE# is\n  generic (\n    BITWIDTH : integer\n  );\n  port "
      "(\n    -- inputs\n    clk          : in std_logic;\n    rst          : "
      "in std_logic;\n    lhs          : in std_logic_vector(BITWIDTH - 1 "
      "downto 0);\n    lhs_valid    : in std_logic;\n    rhs          : in "
      "std_logic_vector(BITWIDTH - 1 downto 0);\n    rhs_valid    : in "
      "std_logic;\n    result_ready : in std_logic;\n    -- outputs\n    "
      "lhs_ready    : out std_logic;\n    rhs_ready    : out std_logic;\n    "
      "result       : out std_logic_vector(BITWIDTH - 1 downto 0);\n    "
      "result_valid : out std_logic);\nend entity;\n\narchitecture arch of "
      "cmpi_#PREDICATE# is\n  signal join_valid : std_logic;\n  signal one     "
      "   : std_logic := \"1\";\n  signal zero       : std_logic := "
      "\"0\";\n\nbegin\n\n  join_write_temp : entity work.join(arch) generic "
      "map(2)\n    port map(\n    (lhs_valid,\n      rhs_valid),\n      "
      "result_ready,\n      join_valid,\n      (lhs_ready,\n      "
      "rhs_ready));\n\n  result <= #CONDTRUE# when (lhs = rhs) else\n    "
      "#CONDFALSE#;\n  result_valid <= join_valid;\nend architecture;\n";
  std::string signed_file =
      "library IEEE;\nuse IEEE.std_logic_1164.all;\nuse "
      "ieee.numeric_std.all;\nuse work.customTypes.all;\n\nentity "
      "cmpi_#PREDICATE# is\n "
      " generic (\n    BITWIDTH : integer\n  );\n  port (\n    -- inputs\n    "
      "clk          : in std_logic;\n    rst          : in std_logic;\n    lhs "
      "         : in std_logic_vector(BITWIDTH - 1 downto 0);\n    lhs_valid   "
      " : in std_logic;\n    rhs          : in std_logic_vector(BITWIDTH - 1 "
      "downto 0);\n    rhs_valid    : in std_logic;\n    result_ready : in "
      "std_logic;\n    -- outputs\n    lhs_ready    : out std_logic;\n    "
      "rhs_ready    : out std_logic;\n    result       : out "
      "std_logic_vector(BITWIDTH - 1 downto 0);\n    result_valid : out "
      "std_logic);\nend entity;\n\narchitecture arch of cmpi_#PREDICATE# is\n  "
      "signal join_valid : std_logic;\n  signal one        : std_logic := "
      "\"1\";\n  signal zero       : std_logic := \"0\";\nbegin\n\n  "
      "join_write_temp : entity work.join(arch) generic map(2)\n    port "
      "map(\n    (lhs_valid,\n      rhs_valid),\n      result_ready,\n      "
      "join_valid,\n      (lhs_ready,\n      rhs_ready));\n\n  result <= one "
      "when (signed(lhs) #TYPEOP# signed(rhs)) else\n    zero;\n  result_valid "
      "<= join_valid;\n\nend architecture;\n";
  std::string unsigned_file =
      "library IEEE;\nuse IEEE.std_logic_1164.all;\nuse "
      "ieee.numeric_std.all;\nuse work.customTypes.all;\n\nentity "
      "cmpi_#PREDICATE# is\n "
      " generic (\n    BITWIDTH : integer\n  );\n  port (\n    -- inputs\n    "
      "clk          : in std_logic;\n    rst          : in std_logic;\n    lhs "
      "         : in std_logic_vector(BITWIDTH - 1 downto 0);\n    lhs_valid   "
      " : in std_logic;\n    rhs          : in std_logic_vector(BITWIDTH - 1 "
      "downto 0);\n    rhs_valid    : in std_logic;\n    result_ready : in "
      "std_logic;\n    -- outputs\n    lhs_ready    : out std_logic;\n    "
      "rhs_ready    : out std_logic;\n    result       : out "
      "std_logic_vector(BITWIDTH - 1 downto 0);\n    result_valid : out "
      "std_logic);\nend entity;\n\narchitecture arch of cmpi_#PREDICATE# is\n  "
      "signal join_valid : std_logic;\n  signal one        : std_logic := "
      "\"1\";\n  signal zero       : std_logic := \"0\";\nbegin\n\n  "
      "join_write_temp : entity work.join(arch) generic map(2)\n    port "
      "map(\n    (lhs_valid,\n      rhs_valid),\n      result_ready,\n      "
      "join_valid,\n      (lhs_ready,\n      rhs_ready));\n\n  result <= one "
      "when (unsigned(lhs) #TYPEOP# unsigned(rhs)) else\n    zero;\n  "
      "result_valid <= join_valid;\n\nend architecture;\n";

  // get the predicate name from command line options
  std::string predicateName(argv[1]);
  std::stringstream buffer;
  std::string modText;
  auto it = comparison.find(predicateName);
  if (it == comparison.end()) {
    auto jt = equality.find(predicateName);
    if (jt != equality.end()) {
      // ne, eq
      buffer << equality_file;
      std::string temp{};
      // replace all #---# sequences
      while (buffer.good()) {
        std::getline(buffer, temp, '#');
        modText += temp;
        if (!buffer.good()) {
          break;
        }
        std::getline(buffer, temp, '#');
        if (temp == "PREDICATE")
          modText += jt->getKey().str();
        else if (temp == "CONDTRUE")
          modText += jt->getValue().first;
        else
          modText += jt->getValue().second;
      }
    } else {
      // wrong predicate
      llvm::errs() << "Wrong predicate\n";
      return 1;
    }
  } else {
    // sge, sgt, sle, slt, uge, ugt, ule, ult
    if (predicateName == "sge" || predicateName == "sgt" ||
        predicateName == "sle" || predicateName == "slt") {
      buffer << signed_file;
    } else
      buffer << unsigned_file;
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
