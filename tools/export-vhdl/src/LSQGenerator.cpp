//===- LSQGenerator.cpp - Generate LSQs -------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LSQGenerator.h"
#include "DOTParser.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#define LSQ_ADDRESSWIDTH_DEFAULT 32
#define LSQ_DATAWIDTH_DEFAULT 32
#define LSQ_FIFODEPTH_DEFAULT 4
#define MAX_BBs 16
#define MAX_LSQ 256

std::ofstream lsqConfigurationFile;
LSQConfiguration lsqConf[MAX_LSQ];
int bbcount;

namespace {
struct BBPort {
  int componendId = -1;
  int loadOffset;
  int loadPort;
  int storeOffset;
  int storePort;
};

struct BBSt {
  int bbcount;
  int bbId;
  BBPort bbid[MAX_BBs];
  int loadPorts;
  int storePorts;
};
} // namespace

BBSt bb[MAX_BBs];

static std::string getLsqName(int lsqIndx) {
  for (int i = 0; i < componentsInNetlist; i++) {
    if (nodes[i].type.find("LSQ") != std::string::npos) {
      if (lsqIndx == nodes[i].lsqIndx)
        return nodes[i].name;
    }
  }
  return "LSQ";
}

static int getLsqFifoDepth(int lsqIndx) {
  for (int i = 0; i < componentsInNetlist; i++) {
    if (nodes[i].type.find("LSQ") != std::string::npos) {
      if (lsqIndx == nodes[i].lsqIndx)
        return nodes[i].fifodepth;
    }
  }
  return 16;
}

// Jiantao, 14/06/2022
static int getLsqFifoLDepth(int lsqIndx) {
  for (int i = 0; i < componentsInNetlist; i++) {
    if (nodes[i].type.find("LSQ") != std::string::npos) {
      if (lsqIndx == nodes[i].lsqIndx) {
        // Jiantao, 05/09/2022
        // Check whether the depths is given or not.
        if (nodes[i].fifodepthL != 0)
          return nodes[i].fifodepthL;
        return nodes[i].fifodepth;
      }
    }
  }
  return 16;
}

static int getLsqFifoSDepth(int lsqIndx) {
  for (int i = 0; i < componentsInNetlist; i++) {
    if (nodes[i].type.find("LSQ") != std::string::npos) {
      if (lsqIndx == nodes[i].lsqIndx) {
        // Jiantao, 05/09/2022
        // Check whether the depths is given or not.
        if (nodes[i].fifodepthS != 0)
          return nodes[i].fifodepthS;
        return nodes[i].fifodepth;
      }
    }
  }
  return 16;
}

static int getLsqLoadPorts(int lsqIndx) {
  for (int i = 0; i < componentsInNetlist; i++) {
    if (nodes[i].type.find("LSQ") != std::string::npos) {
      if (lsqIndx == nodes[i].lsqIndx)
        return nodes[i].loadCount;
    }
  }
  return 0;
}

static int getLsqStorePorts(int lsqIndx) {
  for (int i = 0; i < componentsInNetlist; i++) {
    if (nodes[i].type.find("LSQ") != std::string::npos) {
      if (lsqIndx == nodes[i].lsqIndx)
        return nodes[i].storeCount;
    }
  }
  return 0;
}

static std::string getNumLoads(int lsqIndx) {
  std::string numLoads;
  for (int i = 0; i < componentsInNetlist; i++) {
    if (nodes[i].type.find("LSQ") != std::string::npos) {
      if (lsqIndx == nodes[i].lsqIndx) {
        numLoads = nodes[i].numLoads;
        replace(numLoads.begin(), numLoads.end(), '{', '[');
        replace(numLoads.begin(), numLoads.end(), '}', ']');
        replace(numLoads.begin(), numLoads.end(), ';', ',');
        replace(numLoads.begin(), numLoads.end(), '"', ' ');
        break;
      }
    }
  }

  return numLoads;
}

static std::string getNumStores(int lsqIndx) {
  std::string numStores;
  for (int i = 0; i < componentsInNetlist; i++) {
    if (nodes[i].type.find("LSQ") != std::string::npos) {
      if (lsqIndx == nodes[i].lsqIndx) {
        numStores = nodes[i].numStores;
        replace(numStores.begin(), numStores.end(), '{', '[');
        replace(numStores.begin(), numStores.end(), '}', ']');
        replace(numStores.begin(), numStores.end(), ';', ',');
        replace(numStores.begin(), numStores.end(), '"', ' ');
        break;
      }
    }
  }

  return numStores;
}

static std::string getLoadOffset(int lsqIndx) {
  std::string loadOffset;
  for (int i = 0; i < componentsInNetlist; i++) {
    if (nodes[i].type.find("LSQ") != std::string::npos) {
      if (lsqIndx == nodes[i].lsqIndx) {
        loadOffset = nodes[i].loadOffsets;
        replace(loadOffset.begin(), loadOffset.end(), '{', '[');
        replace(loadOffset.begin(), loadOffset.end(), '}', ']');
        replace(loadOffset.begin(), loadOffset.end(), ';', ',');
        replace(loadOffset.begin(), loadOffset.end(), '"', ' ');
        break;
      }
    }
  }
  return loadOffset;
}

static std::string getStoreOffset(int lsqIndx) {
  std::string storeOffset;
  for (int i = 0; i < componentsInNetlist; i++) {
    if (nodes[i].type.find("LSQ") != std::string::npos) {
      if (lsqIndx == nodes[i].lsqIndx) {
        storeOffset = nodes[i].storeOffsets;
        replace(storeOffset.begin(), storeOffset.end(), '{', '[');
        replace(storeOffset.begin(), storeOffset.end(), '}', ']');
        replace(storeOffset.begin(), storeOffset.end(), ';', ',');
        replace(storeOffset.begin(), storeOffset.end(), '"', ' ');
        break;
      }
    }
  }

  return storeOffset;
}

static std::string getLoadPorts(int lsqIndx) {
  std::string loadPorts;
  for (int i = 0; i < componentsInNetlist; i++) {
    if (nodes[i].type.find("LSQ") != std::string::npos) {
      if (lsqIndx == nodes[i].lsqIndx) {
        loadPorts = nodes[i].loadPorts;
        replace(loadPorts.begin(), loadPorts.end(), '{', '[');
        replace(loadPorts.begin(), loadPorts.end(), '}', ']');
        replace(loadPorts.begin(), loadPorts.end(), ';', ',');
        replace(loadPorts.begin(), loadPorts.end(), '"', ' ');
        break;
      }
    }
  }

  return loadPorts;
}

static std::string getStorePorts(int lsqIndx) {
  std::string storePorts;
  for (int i = 0; i < componentsInNetlist; i++) {
    if (nodes[i].type.find("LSQ") != std::string::npos) {
      if (lsqIndx == nodes[i].lsqIndx) {
        storePorts = nodes[i].storePorts;
        replace(storePorts.begin(), storePorts.end(), '{', '[');
        replace(storePorts.begin(), storePorts.end(), '}', ']');
        replace(storePorts.begin(), storePorts.end(), ';', ',');
        replace(storePorts.begin(), storePorts.end(), '"', ' ');
        break;
      }
    }
  }

  return storePorts;
}

static int getBbindx(int bbId) {
  static int returnIndx = 0;
  for (int indx = 0; indx < MAX_BBs; indx++) {
    if (bb[indx].bbId == bbId) {
      return indx;
    }
    bb[indx].bbId = bbId;
  }
  return returnIndx++;
}

static void mapBb(int lsqIndx) {
  int bbIdindxLoad = 0;
  int bbIdindxStore = 0;
  int bbindx = 0;

  for (int i = 0; i < componentsInNetlist; i++) {
    if (nodes[i].name.find("LSQ") != std::string::npos) {
      if (nodes[i].lsqIndx == lsqIndx) {
        bbcount = nodes[i].bbcount;
      }
    }
    if (nodes[i].bbId != -1) {
      bbindx = getBbindx(nodes[i].bbId);
      if (nodes[i].componentOperator == "lsq_store_op") {
        bb[bbindx].storePorts++;
        bb[bbindx].bbid[bbIdindxStore].storeOffset = nodes[i].offset;
        bb[bbindx].bbid[bbIdindxStore].storePort = nodes[i].portId;
        bb[bbindx].bbid[bbIdindxStore].componendId = i;
        bbIdindxStore++;
      }
      if (nodes[i].componentOperator == "lsq_load_op") {
        bb[bbindx].loadPorts++;
        bb[bbindx].bbid[bbIdindxLoad].loadOffset = nodes[i].offset;
        bb[bbindx].bbid[bbIdindxLoad].loadPort = nodes[i].portId;
        bb[bbindx].bbid[bbIdindxLoad].componendId = i;
        bbIdindxLoad++;
      }
    }
  }
}

static void lsqSetConfiguration(int lsqIndx) {
  mapBb(lsqIndx);
  lsqConf[lsqIndx].name = getLsqName(lsqIndx);
  lsqConf[lsqIndx].dataWidth = getLSQDataWidth();
  lsqConf[lsqIndx].addressWidth = getLSQAddressWidth(lsqIndx);
  lsqConf[lsqIndx].fifoDepth = getLsqFifoDepth(lsqIndx);
  lsqConf[lsqIndx].fifoDepthL = getLsqFifoLDepth(lsqIndx);
  lsqConf[lsqIndx].fifoDepthS = getLsqFifoSDepth(lsqIndx);
  lsqConf[lsqIndx].loadPorts = getLsqLoadPorts(lsqIndx);
  lsqConf[lsqIndx].storePorts = getLsqStorePorts(lsqIndx);

  for (auto &indx : bb) {
    for (int indx2 = 0; indx2 < indx.bbcount; indx2++) {
      lsqConf[lsqIndx].bbParams.loadSizes[indx2] = indx.loadPorts;
      lsqConf[lsqIndx].bbParams.storeSizes[indx2] = indx.storePorts;
    }
  }
}

static std::string getLSQConfigPath(const std::string &outPath, int lsqIndx) {
  return outPath + std::filesystem::path::preferred_separator + "lsq" +
         std::to_string(lsqIndx) + "_config.json";
  ;
}

static void lsqWriteConfigurationFile(const std::string &outPath, int lsqIndx) {
  lsqConfigurationFile.open(getLSQConfigPath(outPath, lsqIndx));

  lsqConfigurationFile << "{\n";

  lsqConfigurationFile << R"("name": ")" << lsqConf[lsqIndx].name << "\",\n";
  lsqConfigurationFile << "\"dataWidth\":" << lsqConf[lsqIndx].dataWidth
                       << ",\n";
  lsqConfigurationFile << R"("experimental" : false )"
                       << ",\n";
  lsqConfigurationFile << R"("accessType" : "BRAM" )"
                       << ",\n";
  lsqConfigurationFile << "\"addrWidth\":" << lsqConf[lsqIndx].addressWidth
                       << ",\n";
  lsqConfigurationFile << "\"fifoDepth\":" << lsqConf[lsqIndx].fifoDepth
                       << ",\n";
  lsqConfigurationFile << "\"fifoDepth_L\":" << lsqConf[lsqIndx].fifoDepthL
                       << ",\n";
  lsqConfigurationFile << "\"fifoDepth_S\":" << lsqConf[lsqIndx].fifoDepthS
                       << ",\n";
  lsqConfigurationFile << "\"numLoadPorts\":" << lsqConf[lsqIndx].loadPorts
                       << ",\n";
  lsqConfigurationFile << "\"numStorePorts\":" << lsqConf[lsqIndx].storePorts
                       << ",\n";
  lsqConfigurationFile << "\"numBBs\": " << bbcount << ",\n";
  lsqConfigurationFile << "\"numLoads\": " << getNumLoads(lsqIndx) << ",\n";
  lsqConfigurationFile << "\"numStores\": " << getNumStores(lsqIndx) << ",\n";
  lsqConfigurationFile << "\"loadOffsets\": " << getLoadOffset(lsqIndx)
                       << ",\n";
  lsqConfigurationFile << "\"storeOffsets\": " << getStoreOffset(lsqIndx)
                       << ",\n";
  lsqConfigurationFile << "\"loadPorts\": " << getLoadPorts(lsqIndx) << ",\n";
  lsqConfigurationFile << "\"storePorts\": " << getStorePorts(lsqIndx) << ",\n";

  lsqConfigurationFile << "\"bufferDepth\": 0\n";

  lsqConfigurationFile << "}\n";

  lsqConfigurationFile.close();
}

void lsqGenerateConfiguration(const std::string &outPath) {
  for (int lsqIndx = 0; lsqIndx < lsqsInNetlist; lsqIndx++) {
    lsqSetConfiguration(lsqIndx);
    lsqWriteConfigurationFile(outPath, lsqIndx);
  }
}

int getLSQDataWidth() {
  for (int i = 0; i < componentsInNetlist; i++) {
    if (nodes[i].name.find("load") != std::string::npos)
      return nodes[i].inputs.input[0].bitSize;
  }
  return LSQ_DATAWIDTH_DEFAULT;
}

int getLSQAddressWidth(int lsqIndx) {
  for (int i = 0; i < componentsInNetlist; i++) {
    if (nodes[i].type.find("LSQ") != std::string::npos) {
      if (lsqIndx == nodes[i].lsqIndx)
        return nodes[i].addressSize;
    }
  }
  return LSQ_ADDRESSWIDTH_DEFAULT;
}

// JSON Example with three components
// {
//   "specifications"  :[
//   {
//     "name": "hist",
//     "dataWidth": 32,
//     "addressWidth": 10,
//     "fifoDepth": 4,
//     "loadPorts": 1,
//     "storePorts": 1,
//     "bbParams": {
//     "loadSizes": [1],
//     "storeSizes": [1],
//     "loadOffsets": [[0, 0, 0, 0]],
//     "storeOffsets": [[1, 0, 0, 0]],
//     "loadPortsList": [[0, 0, 0, 1]],
//     "storePortsList": [[0, 0, 0, 0]]
//     }
//   },
//   {
//     "name": "hist",
//     "dataWidth": 32,
//     "addressWidth": 10,
//     "fifoDepth": 8,
//     "loadPorts": 1,
//     "storePorts": 1,
//     "bbParams": {
//     "loadSizes": [1],
//     "storeSizes": [1],
//     "loadOffsets": [[0, 0, 0, 0, 0, 0, 0, 0]],
//     "storeOffsets": [[1, 0, 0, 0, 0, 0, 0, 0]],
//     "loadPortsList": [[0, 0, 0, 0, 0, 0, 0, 1]],
//     "storePortsList": [[0, 0, 0, 0, 0, 0, 0, 0]]
//     }
//   },
//   {
//     "name": "hist",
//     "dataWidth": 32,
//     "addressWidth": 10,
//     "fifoDepth": 16,
//     "loadPorts": 1,
//     "storePorts": 1,
//     "bbParams": {
//     "loadSizes": [1],
//     "storeSizes": [1],
//     "loadOffsets": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
//     "storeOffsets": [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
//     "loadPortsList": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
//     "storePortsList": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
//     }
//   }
//   ]
// }
