//===- constant.cpp - Fulfill a constant generator --------------*- C++ -*-===//
//
// Experimental tool that realizes a generator for constant component
// (handshake.constant)
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <bitset>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace llvm;

static const std::string CONSTANT_PATH =
    "experimental/data/vhdl/handshake/constant.vhd";

int main(int argc, char **argv) {
  // no value provided
  if (argc < 2) {
    llvm::errs() << "Too few arguments in generator call\n";
    return 1;
  }
  // read as file
  std::ifstream file;
  file.open(CONSTANT_PATH);

  if (!file.is_open()) {
    llvm::errs() << "Filepath is uncorrect\n";
    return 1;
  }
  // get the constant value from command line options
  std::string constantStr(argv[1]);
  // get the bitwidth from command line options
  std::string bitwidthStr(argv[2]);
  size_t bitwidth = std::atoi(bitwidthStr.c_str());
  size_t num = std::atoi(constantStr.c_str());
  std::stringstream buffer;
  std::string modText;
  buffer << file.rdbuf();
  std::string temp;
  // replace all #---# sequences
  while (buffer.good()) {
    std::getline(buffer, temp, '#');
    modText += temp;
    if (!buffer.good())
      break;
    std::getline(buffer, temp, '#');
    if (temp == "CST_VALUE") {
      // convert to binary...
      std::bitset<32> x(num);
      auto toBinary = x.to_string();
      // ... and make sure, that number of bits = bitwidth
      modText += "\"" + toBinary.substr(32 - bitwidth) + "\"";
    } else if (temp == "CST_NAME")
      modText += constantStr.c_str();
  }

  // print the result module text to std output
  llvm::outs() << modText;
  return 0;
}
