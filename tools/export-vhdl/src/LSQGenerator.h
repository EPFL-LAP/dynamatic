//===- LSQGenerator.h - Generate LSQs ---------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EXPORT_VHDL_LSQ_GENERATOR_H
#define EXPORT_VHDL_LSQ_GENERATOR_H

#include <string>

#define MAX_SIZES 16

void lsqGenerateConfiguration(const std::string &topLevelFilename);
void lsqGenerate(const std::string &topLevelFilename);

int getLSQDataWidth();
int getLSQAddressWidth(int lsqIndx);

struct BBParams {
  int loadSizes[MAX_SIZES];      //     "loadSizes": [1],
  int storeSizes[MAX_SIZES];     //     "storeSizes": [1],
  int loadOffsets[MAX_SIZES];    //     "loadOffsets": [[0, 0, 0, 0]],
  int storeOffsets[MAX_SIZES];   //     "storeOffsets": [[1, 0, 0, 0]],
  int loadPortsList[MAX_SIZES];  //     "loadPortsList": [[0, 0, 0, 1]],
  int storePortsList[MAX_SIZES]; //     "storePortsList": [[0, 0, 0, 0]]
};

struct LSQConfiguration {
  std::string name;  //     "name": "hist",
  int dataWidth;     //     "dataWidth": 32,
  int addressWidth;  //     "addressWidth": 10,
  int fifoDepth;     //     "fifoDepth": 4,
  int fifoDepthL;    //     "fifoDepth_L": 4,
  int fifoDepthS;    //     "fifoDepth_S": 4,
  int loadPorts;     //     "loadPorts": 1,
  int storePorts;    //     "storePorts": 1,
  BBParams bbParams; //     "bbParams": {
};

#endif // EXPORT_VHDL_LSQ_GENERATOR_H
