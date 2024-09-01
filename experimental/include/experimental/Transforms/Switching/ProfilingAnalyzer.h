//===- SwitchingEstimation.h - Switching Estimation -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Analyzer Class to analyze the functional 
// profiling result (Both Data Trace and the BB list).
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_TRANSFORMS_PROFILING_ANALYZER_H
#define EXPERIMENTAL_TRANSFORMS_PROFILING_ANALYZER_H

#include "experimental/Transforms/Switching/SwitchingSupport.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/Attribute.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <set>
#include <sstream>
#include <filesystem>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

//===----------------------------------------------------------------------===//
//
//  Helper Class
//
//===----------------------------------------------------------------------===//

class SCFFile {
public:
  SCFFile(StringRef scfFile);

  // Vector storing all SCF level op names in the mlir file
  std::vector<std::string> opNameList;

};

//===----------------------------------------------------------------------===//
//
//  Profiler Analyzer Class
//
//===----------------------------------------------------------------------===//

// Class used to analyze the scf-level software profiling
// 
class SCFProfilingResult {
public:

  SCFProfilingResult(StringRef dataTrace,
                     StringRef bbList,
                     SwitchingInfo& switchInfo);

  // Parse the BB execution list log file
  void parseBBListFile(StringRef bbList, SwitchingInfo& switchInfo);

  // Parse the actual data profiling log file
  void parseDataLogFile(StringRef dataTrace, SwitchingInfo& switchInfo);

  // Construct the map from scf level profiling to handshake level IR
  void buildScfToHSMap(SwitchingInfo& switchInfo, SCFFile& scfFile);

  // This function insert new (value, iter index) pair to the value list
  void insertValuePair(int opValue, unsigned iterIndex, std::string opName);

  // This function constructs the overall segment execution counts
  void constructSegExeCount();

  //
  //  Global Storing Structure for analyzing the profiling results
  //
  std::vector<unsigned> executedBBTrace;                      // Vector used to store the execution trace of BB labels
  std::vector<unsigned> iterEndIndex;                         // Vector storing the ending edge index on the boundary of different segments
  std::vector<std::string> executedSegTrace;                  // Vector used to store the execution trace consists of segment labels
  std::map<unsigned, unsigned> bbToIterMap;                   // Map from BB index to the corresponding Iteration index
  std::map<unsigned, std::pair<std::string, unsigned>> execPhaseToSegExecNumMap;     // Map from execution stage to (segLabel, numExec) pair
  std::map<std::string, std::vector<std::pair<int, unsigned>>> opNameToValueListMap; // Map from Hndshake level opName to the list of value in the data profiling process
  
  // Map from SCF level op name to Handshake level op name
  std::map<std::string, std::string> scfToHandshakeNameMap;

};


#endif // EXPERIMENTAL_TRANSFORMS_PROFILING_ANALYZER_H


