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

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

// Class used to analyze the scf-level software profiling
// 
class SCFProfilingResult {
  SCFProfilingResult(StringRef dataTrace,
                     StringRef bbList,
                     SwitchingInfo& switchInfo);

  // Parse the BB execution list log file
  void parseBBListFile(StringRef bbList, SwitchingInfo& switchInfo);

  // Parse the actual data profiling log file
  void parseDataLogFile(StringRef dataTrace, SwitchingInfo& switchInfo );

  //
  //  Global Storing Structure for analyzing the profiling results
  //
  std::vector<std::string> executed_bb_trace;   // Vector used to store the execution trace of BB labels
  std::vector<int> iter_end_index;              // Vector storing the ending edge index on the boundary of different segments
  std::vector<std::string> execution_trace;     // Vector used to store the execution trace consists of segment labels


  
};


#endif // EXPERIMENTAL_TRANSFORMS_PROFILING_ANALYZER_H


