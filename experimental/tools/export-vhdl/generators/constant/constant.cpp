//===- constant.cpp - Fulfill a constant generator --------------*- C++ -*-===//
//
// Experimental tool that realizes a generator for constant component
// (handshake.constant)
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
  // no value provided
  if (argc < 2) {
    llvm::errs() << "Too few arguments in generator call\n";
    return 1;
  }

  // necessary file
  std::string const_file =
      "library ieee;\nuse ieee.std_logic_1164.all;\nuse "
      "ieee.numeric_std.all;\nuse work.customTypes.all;\n\nentity constant "
      "is\n  generic (\n    BITWIDTH : integer\n  );\n  port (\n    -- "
      "inputs\n    clk          : in std_logic;\n    rst          : in "
      "std_logic;\n    ctrl_valid   : in std_logic;\n    result_ready : in "
      "std_logic;\n    -- outputs\n    ctrl_ready   : out std_logic;\n    "
      "result       : out std_logic_vector(BITWIDTH - 1 downto 0);\n    "
      "result_valid : out std_logic);\nend constant;\n\narchitecture arch of "
      "constant is\nbegin\n  result       <= #CST_VALUE#;\n  result_valid <= "
      "ctrl_valid;\n  ctrl_ready   <= result_ready;\nend architecture;\n";

  // get the predicate name from command line options
  std::string predicateName(argv[1]);
  std::stringstream buffer;
  std::string modText;
  buffer << const_file;
  std::string temp{};
  // replace all #---# sequences
  while (buffer.good()) {
    std::getline(buffer, temp, '#');
    modText += temp;
    if (!buffer.good())
      break;

    std::getline(buffer, temp, '#');
    if (temp == "CST_VALUE")
      modText += predicateName;
  }

  // print the result module text to std output
  llvm::outs() << modText;
  return 0;
}
