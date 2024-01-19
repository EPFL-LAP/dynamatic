//===- DOTParser.h - Parse DOT file -----------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EXPORT_VHDL_DOT_PARSER_H
#define EXPORT_VHDL_DOT_PARSER_H

#include <string>
#include <vector>

#define COMMENT_CHARACTER '/'
#define MAX_INPUTS 512
#define MAX_OUTPUTS 512
#define COMPONENT_NOT_FOUND -1
#define MAX_NODES 16384

struct Input {
  int bitSize = 32;
  int prevNodesID = COMPONENT_NOT_FOUND;
  std::string type;
  int port;
  std::string infoType;
};

struct In {
  int size = 0;
  Input input[MAX_OUTPUTS];
};

struct Output {
  int bitSize = 32;
  int nextNodesID = COMPONENT_NOT_FOUND;
  int nextNodesPort;
  std::string type;
  int port;
  std::string infoType;
};

struct Out {
  int size = 0;
  Output output[MAX_OUTPUTS];
};

struct Node {
  std::string name;
  std::string type;
  In inputs;
  Out outputs;
  std::string parameters;
  int nodeId;
  int componentType;
  std::string componentOperator;
  unsigned long int componentValue;
  bool componentControl;
  int slots;
  bool trasparent;
  std::string memory;
  int bbcount = -1;
  int loadCount = -1;
  int storeCount = -1;
  int dataSize = 32;
  int addressSize = 32;
  bool memAddress;
  int bbId = -1;
  int portId = -1;
  int offset = -1;
  int lsqIndx = -1;
  // for sel
  std::vector<std::vector<int>> orderings;

  std::string numLoads;
  std::string numStores;
  std::string loadOffsets;
  std::string storeOffsets;
  std::string loadPorts;
  std::string storePorts;

  // Jiantao, 14/06/2022
  int fifodepthL;
  int fifodepthS;
  int fifodepth;
  int constants;
};

void parseDOT(const std::string &filename);

extern Node nodes[MAX_NODES];
extern int componentsInNetlist;
extern int lsqsInNetlist;

#endif // EXPORT_VHDL_DOT_PARSER_H
