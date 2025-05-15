//===- HlsVhdlTb.h ----------------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HLS_VERIFIER_HLS_VHDL_TB_H
#define HLS_VERIFIER_HLS_VHDL_TB_H

#include "CAnalyser.h"
#include "VerificationContext.h"
#include "mlir/Support/IndentedOstream.h"
#include <string>
#include <vector>

using namespace std;

namespace hls_verify {

class MemElem {
public:
  bool isArray;

  string ce0SignalName;
  string we0SignalName;
  string dIn0SignalName;
  string dOut0SignalName;
  string addr0SignalName;
  string ce1SignalName;
  string we1SignalName;
  string dIn1SignalName;
  string dOut1SignalName;
  string addr1SignalName;
  string inFileParamValue;
  string outFileParamValue;
  string dataWidthParamValue;
  string addrWidthParamValue;
  string dataDepthParamValue;

  string memStartSignalName;
  string memEndSignalName;
};

class Constant {
public:
  string constName;
  string constType;
  string constValue;
  Constant(const string &name, const string &type, const string &value);
};

class HlsVhdlTb {
public:
  HlsVhdlTb(const VerificationContext &ctx);
  void generateVhdlTestbench(mlir::raw_indented_ostream &os);
  string getInputFilepathForParam(const CFunctionParameter &param);
  string getOutputFilepathForParam(const CFunctionParameter &param);

private:
  VerificationContext ctx;
  string duvName;
  string tleName;
  vector<CFunctionParameter> cDuvParams;
  vector<pair<string, string>> duvPortMap;
  vector<Constant> constants;
  vector<MemElem> memElems;

  void getConstantDeclaration(mlir::raw_indented_ostream &os);
  void getSignalDeclaration(mlir::raw_indented_ostream &os);
  void getMemoryInstanceGeneration(mlir::raw_indented_ostream &os);
  void getDuvInstanceGeneration(mlir::raw_indented_ostream &os);
  void getDuvComponentDeclaration(mlir::raw_indented_ostream &os);
  void getOutputTagGeneration(mlir::raw_indented_ostream &os);
  int getTransactionNumberFromInput();
};
} // namespace hls_verify

#endif // HLS_VERIFIER_HLS_VHDL_TB_H
