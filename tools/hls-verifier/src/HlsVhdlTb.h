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
#include <string>
#include <vector>

using namespace std;

namespace hls_verify {

class MemElem {
public:
  bool isArray;

  static string clkPortName;
  static string rstPortName;
  static string ce0PortName;
  static string we0PortName;
  static string dIn0PortName;
  static string dOut0PortName;
  static string addr0PortName;
  static string ce1PortName;
  static string we1PortName;
  static string dIn1PortName;
  static string dOut1PortName;
  static string addr1PortName;
  static string donePortName;
  static string inFileParamName;
  static string outFileParamName;
  static string dataWidthParamName;
  static string addrWidthParamName;
  static string dataDepthParamName;

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
  string generateVhdlTestbench();
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

  string getLibraryHeader();
  string getEntitiyDeclaration();
  string getArchitectureBegin();
  string getConstantDeclaration();
  string getSignalDeclaration();
  string getMemoryInstanceGeneration();
  string getDuvInstanceGeneration();
  string getDuvComponentDeclaration();
  string getCommonBody();
  string getArchitectureEnd();
  string getOutputTagGeneration();
  int getTransactionNumberFromInput();

  static string getCe0PortNameForCParam(string &cParam);
  static string getWe0PortNameForCParam(string &cParam);
  static string getDataIn0PortNameForCParam(string &cParam);
  static string getDataOut0PortNameForCParam(string &cParam);
  static string getAddr0PortNameForCParam(string &cParam);
  static string getCe1PortNameForCParam(string &cParam);
  static string getWe1PortNameForCParam(string &cParam);
  static string getDataIn1PortNameForCParam(string &cParam);
  static string getDataOut1PortNameForCParam(string &cParam);
  static string getAddr1PortNameForCParam(string &cParam);
  static string getReadyInPortNameForCParam(string &cParam);
  static string getReadyOutPortNameForCParam(string &cParam);
  static string getValidInPortNameForCParam(string &cParam);
  static string getValidOutPortNameForCParam(string &cParam);
  static string getDataInSaPortNameForCParam(string &cParam);
  static string getDataOutSaPortNameForCParam(string &cParam);
};
} // namespace hls_verify

#endif // HLS_VERIFIER_HLS_VHDL_TB_H
