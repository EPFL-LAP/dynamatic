//===- DOTParser.cpp - Parse DOT file ---------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DOTParser.h"
#include "StringUtils.h"
#include "VHDLWriter.h"
#include "assert.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>

using std::vector;

using namespace std;

Node nodes[MAX_NODES];

int componentsInNetlist;
int lsqsInNetlist;

/// This is certifiably cursed!
///
/// It is also the easiest way to replicate dot2vhdl's behavior which uses
/// exception handling around calls to `std::stoi` to handle arguments with
/// trailing garbage characters. We do not use exceptions, hence this
/// monstrosity to ensure that calls to `std::stoi` always succeed.
///
/// Thanks legacy!
static int stoiSubstr(const std::string &str) {
  size_t idx = 0;
  while (idx < str.size() && std::isdigit(str[idx]))
    ++idx;
  return idx == 0 ? 0 : std::stoi(str.substr(0, idx));
}

static bool isConnection(const string &line) {
  if (line.find("type") != std::string::npos)
    return false;

  if (line.find(">") != std::string::npos)
    return true;
  std::cerr << "Cannot parse DOT line:\n" << line << std::endl;
  exit(1);
}

static string getValue(const string &parameter) {
  vector<string> v;
  stringSplit(parameter, '=', v);
  if (!v.empty())
    return v[1];
  return "";
}

static string getComponentType(string parameters) {

  parameters = stringClean(parameters);

  string type = getValue(parameters);
  nodes[componentsInNetlist].componentType = COMPONENT_GENERIC;

  return type;
}

static string getComponentOperator(string parameters) {
  parameters = stringClean(parameters);

  string type = getValue(parameters);

  return type;
}

static string getComponentValue(string parameters) {
  parameters = stringClean(parameters);

  string type = getValue(parameters);
  return type;
}

static bool getComponentControl(string parameters) {
  parameters = stringClean(parameters);

  string type = getValue(parameters);

  return type == "true";
}

static int getComponentSlots(string parameters) {

  parameters = stringClean(parameters);

  string type = getValue(parameters);
  return stoiSubstr(type);
}

static bool getComponentTransparent(string parameters) {
  parameters = stringClean(parameters);

  string type = getValue(parameters);

  return type == "true";
}

static string getComponentMemory(string parameters) {
  parameters = stringClean(parameters);

  string type = getValue(parameters);
  return type;
}

static string getComponentNumloads(const string &parameters) {
  string type = getValue(parameters);
  return type;
}

static string getComponentNumstores(const string &parameters) {
  // parameters = stringClean( parameters );

  string type = getValue(parameters);
  return type;
}

static int getComponentBbcount(string parameters) {
  parameters = stringClean(parameters);

  string type = getValue(parameters);
  return stoiSubstr(type);
}

static int getComponentBbId(string parameters) {
  parameters = stringClean(parameters);

  string type = getValue(parameters);
  return stoiSubstr(type);
}

static int getComponentPortId(string parameters) {
  parameters = stringClean(parameters);

  string type = getValue(parameters);
  return stoiSubstr(type);
}

static int getComponentOffset(string parameters) {
  parameters = stringClean(parameters);

  string type = getValue(parameters);
  return stoiSubstr(type);
}

static bool getComponentMemAddress(string parameters) {
  parameters = stringClean(parameters);

  string type = getValue(parameters);

  return type == "true";
}

static int getComponentConstants(string parameters) {
  parameters = stringClean(parameters);

  string type = getValue(parameters);
  return stoiSubstr(type);
}

static vector<vector<int>> getComponentOrderings(const string &parameter) {
  vector<vector<int>> orderings;
  vector<string> par = vector<string>();
  stringSplit(parameter, '=', par);
  assert(par.size() == 2);
  par[1].erase(remove(par[1].begin(), par[1].end(), '"'), par[1].end());
  string value = par[1];
  // trim the value
  int startIndex = value.find_first_not_of(" ");
  int endIndex = value.find_last_not_of(" ") + 1;
  value = value.substr(startIndex, endIndex);

  vector<string> orderingPerBb = vector<string>();
  if (value.find(" ") != string::npos) {
    stringSplit(value, ' ', orderingPerBb);
  } else {
    orderingPerBb.push_back(value);
  }

  for (const auto &orderingInsideBb : orderingPerBb) {
    vector<string> stringIndices{};
    if (orderingInsideBb.find("|") != string::npos) {
      stringSplit(orderingInsideBb, '|', stringIndices);
    } else {
      stringIndices.push_back(orderingInsideBb);
    }
    vector<int> intIndices{};
    intIndices.reserve(stringIndices.size());
    for (const auto &stringIndex : stringIndices) {
      intIndices.push_back(stoiSubstr(stringIndex));
    }
    orderings.push_back(intIndices);
  }
  return orderings;
}

static string getInputType(const string &in) {
  vector<string> par;
  string retVal = "u";

  stringSplit(in, '*', par);

  if (!par.empty()) {
    par[1].erase(remove(par[1].begin(), par[1].end(), '"'), par[1].end());
    retVal = par[1].at(0);
  }

  return retVal;
}

static int getInputPort(const string &in) {
  vector<string> par;
  int retVal = 0;
  string val;

  stringSplit(in, '*', par);

  if (!par.empty()) {
    par[1].erase(remove(par[1].begin(), par[1].end(), '"'), par[1].end());
    if (par[1].size() > 1) {
      val = par[1].at(1);
      retVal = stoiSubstr(val);
    }
  }
  return retVal;
}

static string getInfoType(const string &in) {
  vector<string> par;
  string retVal = "u";

  stringSplit(in, '*', par);

  if (!par.empty()) {
    par[1].erase(remove(par[1].begin(), par[1].end(), '"'), par[1].end());
    if (par[1].size() > 2) {
      retVal = par[1].at(2);
    }
  }

  return retVal;
}

static int getInputSize(const string &in) {

  vector<string> bitSizes;
  int retVal = 32;

  stringSplit(in, ':', bitSizes);

  if (!bitSizes.empty()) {
    bitSizes[1].erase(remove(bitSizes[1].begin(), bitSizes[1].end(), '"'),
                      bitSizes[1].end());
    bitSizes[1].erase(remove(bitSizes[1].begin(), bitSizes[1].end(), ']'),
                      bitSizes[1].end());
    bitSizes[1].erase(remove(bitSizes[1].begin(), bitSizes[1].end(), ';'),
                      bitSizes[1].end());

    if (stoi(bitSizes[1]) == 0)
      retVal = 1;
    else
      retVal = stoi(bitSizes[1]);
  }

  return retVal;
}

static In getComponentInputs(string in, int componentsInNetlist) {
  vector<string> v;
  vector<string> par;
  vector<string> bitSizes;
  In inputs;

  in.erase(remove(in.begin(), in.end(), '\t'), in.end());

  inputs.size = 0;

  stringSplit(in, '=', par);

  if (!par.empty()) {
    par[1].erase(remove(par[1].begin(), par[1].end(), '"'), par[1].end());
  }

  string testString;

  for (int indx = MAX_INPUTS; indx > 0; indx--) {
    testString = "in";
    testString += to_string(indx);
    if (par[1].find(testString) != std::string::npos) {
      inputs.size = indx;
      break;
    }
  }

  int inputIndx = 0;

  if (inputs.size == 1) {
    inputs.input[inputIndx].bitSize = getInputSize(par[1]);

    inputs.input[inputIndx].type = getInputType(par[1]);
    inputs.input[inputIndx].port = getInputPort(par[1]);
    inputs.input[inputIndx].infoType = getInfoType(par[1]);
    if (inputs.input[inputIndx].infoType == "a") {
      nodes[componentsInNetlist].addressSize = inputs.input[inputIndx].bitSize;
    }
    if (inputs.input[inputIndx].infoType == "d") {
      nodes[componentsInNetlist].dataSize = inputs.input[inputIndx].bitSize;
    }

  } else {
    stringSplit(par[1], ' ', v);

    if (!v.empty()) {
      for (const auto &indx : v) {
        if (!(indx.empty())) {
          inputs.input[inputIndx].bitSize = getInputSize(indx);
          inputs.input[inputIndx].type = getInputType(indx);
          inputs.input[inputIndx].port = getInputPort(indx);
          inputs.input[inputIndx].infoType = getInfoType(indx);
          if (inputs.input[inputIndx].infoType == "a") {
            nodes[componentsInNetlist].addressSize =
                inputs.input[inputIndx].bitSize;
          }
          if (inputs.input[inputIndx].infoType == "d") {
            nodes[componentsInNetlist].dataSize =
                inputs.input[inputIndx].bitSize;
          }
          inputIndx++;
        }
      }
    }
  }

  return inputs;
}

static Out getComponentOutputs(const string &parameters) {
  vector<string> v;
  vector<string> bitSizes;
  vector<string> par;

  Out outputs;

  outputs.size = 0;

  stringSplit(parameters, '=', par);

  if (!par.empty()) {
    par[1].erase(remove(par[1].begin(), par[1].end(), '"'), par[1].end());
  }

  string testString;

  for (int indx = MAX_OUTPUTS; indx > 0; indx--) {
    testString = "out";
    testString += to_string(indx);
    if (parameters.find(testString) != std::string::npos) {
      outputs.size = indx;
      break;
    }
  }

  int outputIndx = 0;

  if (outputs.size == 1) {
    outputs.output[outputIndx].bitSize = getInputSize(par[1]);
    outputs.output[outputIndx].type = getInputType(par[1]);
    outputs.output[outputIndx].port = getInputPort(par[1]);
    outputs.output[outputIndx].infoType = getInfoType(par[1]);
  } else {
    stringSplit(par[1], ' ', v);
    if (!v.empty()) {
      for (const auto &indx : v) {
        if (!(indx.empty())) {
          outputs.output[outputIndx].bitSize = getInputSize(indx);
          outputs.output[outputIndx].type = getInputType(indx);
          outputs.output[outputIndx].port = getInputPort(indx);
          outputs.output[outputIndx].infoType = getInfoType(indx);
          outputIndx++;
        }
      }
    }
  }

  return outputs;
}

static string getComponentName(string name) {
  string nameRet;
  name.erase(remove(name.begin(), name.end(), '\t'), name.end());
  name.erase(remove(name.begin(), name.end(), '"'), name.end());
  name.erase(remove(name.begin(), name.end(), ' '), name.end());

  if (name[0] == '_')
    name.replace(0, 1, "");

  nameRet = name;
  return nameRet;
}

static void parseConnections(const string &line) {
  vector<string> v;
  vector<string> fromTo;
  vector<string> parameters;

  int currentNodeId;
  int nextNodeId;

  stringSplit(line, '>', v);

  int i;
  if (!v.empty()) {
    v[0].erase(remove(v[0].begin(), v[0].end(), ' '), v[0].end());
    v[0].erase(remove(v[0].begin(), v[0].end(), '-'), v[0].end());
    v[0].erase(remove(v[0].begin(), v[0].end(), '\t'), v[0].end());
    v[0].erase(remove(v[0].begin(), v[0].end(), '"'), v[0].end());

    stringSplit(v[1], '[', fromTo);
    fromTo[0].erase(remove(fromTo[0].begin(), fromTo[0].end(), ' '),
                    fromTo[0].end());
    fromTo[0].erase(remove(fromTo[0].begin(), fromTo[0].end(), '\t'),
                    fromTo[0].end());
    fromTo[0].erase(remove(fromTo[0].begin(), fromTo[0].end(), '"'),
                    fromTo[0].end());

    if (v[0][0] == '_') {
      v[0].replace(0, 1, "");
    }

    if (fromTo[0][0] == '_') {
      fromTo[0].replace(0, 1, "");
    }

    currentNodeId = COMPONENT_NOT_FOUND;
    for (i = 0; i < componentsInNetlist; i++) {
      if (nodes[i].name.compare(v[0]) == 0) {
        currentNodeId = i;
        break;
      }
    }
    nextNodeId = COMPONENT_NOT_FOUND;

    for (i = 0; i < componentsInNetlist; i++) {
      if (nodes[i].name.compare(fromTo[0]) == 0) {
        nextNodeId = i;
        break;
      }
    }

    stringSplit(fromTo[1], ',', parameters);

    int inputIndx;
    int outputIndx;
    for (auto &parameter : parameters) {
      if (parameter.find("from") != std::string::npos) {
        parameter.erase(remove(parameter.begin(), parameter.end(), ' '),
                        parameter.end());
        parameter.erase(remove(parameter.begin(), parameter.end(), '\t'),
                        parameter.end());
        parameter.erase(remove(parameter.begin(), parameter.end(), '"'),
                        parameter.end());
        parameter.erase(0, 8);
        outputIndx = stoiSubstr(parameter);
        outputIndx--;
      }
      if (parameter.find("to") != std::string::npos) {
        parameter.erase(remove(parameter.begin(), parameter.end(), ' '),
                        parameter.end());
        parameter.erase(remove(parameter.begin(), parameter.end(), '\t'),
                        parameter.end());
        parameter.erase(remove(parameter.begin(), parameter.end(), '"'),
                        parameter.end());
        parameter.erase(remove(parameter.begin(), parameter.end(), ';'),
                        parameter.end());
        parameter.erase(remove(parameter.begin(), parameter.end(), ']'),
                        parameter.end());

        parameter.erase(0, 5);

        inputIndx = stoiSubstr(parameter);
        inputIndx--;
      }
    }

    if (currentNodeId != COMPONENT_NOT_FOUND &&
        nextNodeId != COMPONENT_NOT_FOUND) {

      nodes[currentNodeId].outputs.output[outputIndx].nextNodesID = nextNodeId;
      nodes[currentNodeId].outputs.output[outputIndx].nextNodesPort = inputIndx;
      nodes[nextNodeId].inputs.input[inputIndx].prevNodesID = currentNodeId;

    } else {
      cerr << "Netlist Error" << endl;

      if (currentNodeId == COMPONENT_NOT_FOUND) {
        cerr << "Node Description " << v[0] << " not found. Not ID assigned"
             << endl;
      } else

          if (nextNodeId == COMPONENT_NOT_FOUND) {
        cerr << "Node ID" << currentNodeId
             << "Node Name: " << nodes[currentNodeId].name
             << " has not next node for output " << outputIndx << endl;
      }

      cerr << "Exiting without producing netlist" << endl;
      exit(0);
    }
  }
}

static string checkComments(string line) {
  vector<string> v;
  stringSplit(line, COMMENT_CHARACTER, v);

  if (!v.empty())
    return v[0];
  return line;
}

static void parseComponents(const string &v0, const string &v1) {
  vector<string> parameters;
  string parameter;

  nodes[componentsInNetlist].name = getComponentName(v0);
  if (!(nodes[componentsInNetlist].name.empty())) {
    stringSplit(v1, ',', parameters);
    for (const auto &indx : parameters) {
      parameter = stringRemoveBlank(indx);

      if (parameter.find("type") != std::string::npos) {
        nodes[componentsInNetlist].type = getComponentType(indx);
        nodes[componentsInNetlist].componentOperator =
            nodes[componentsInNetlist].type; // For the component without an
                                             // operator, sets the entity type
        if (nodes[componentsInNetlist].type == "LSQ") {
          nodes[componentsInNetlist].lsqIndx = lsqsInNetlist;
          lsqsInNetlist++;
        }
      }
      if (parameter.find("in=") != std::string::npos) {
        nodes[componentsInNetlist].inputs =
            getComponentInputs(indx, componentsInNetlist);
      }
      if (parameter.find("out=") != std::string::npos) {
        nodes[componentsInNetlist].outputs = getComponentOutputs(indx);
      }
      if (parameter.find("op") != std::string::npos) {
        nodes[componentsInNetlist].componentOperator =
            getComponentOperator(indx);
      }
      if (parameter.find("value") != std::string::npos) {
        // nodes[componentsInNetlist].componentValue = protected_stoi(
        // get_component_value ( parameters[indx] ) );
        unsigned long int hexValue;
        hexValue = strtoul(getComponentValue(indx).c_str(), nullptr, 16);
        nodes[componentsInNetlist].componentValue = hexValue;
      }
      if (parameter.find("control") != std::string::npos) {
        nodes[componentsInNetlist].componentControl = getComponentControl(indx);
      }
      if (parameter.find("slots") != std::string::npos) {
        nodes[componentsInNetlist].slots = getComponentSlots(indx);

        switch (nodes[componentsInNetlist].slots) {
        case 1:
        case 2:
        case 0:
          break;
        default:
          nodes[componentsInNetlist].type = "Fifo";
          // For the component without an operator, sets the entity type
          nodes[componentsInNetlist].componentOperator =
              nodes[componentsInNetlist].type;
        }
      }
      if (parameter.find("transparent") != std::string::npos) {
        nodes[componentsInNetlist].trasparent = getComponentTransparent(indx);
      }
      if (parameter.find("memory") != std::string::npos) {
        nodes[componentsInNetlist].memory = getComponentMemory(indx);
      }
      if (parameter.find("bbcount") != std::string::npos) {
        nodes[componentsInNetlist].bbcount = getComponentBbcount(indx);

        if (nodes[componentsInNetlist].bbcount == 0) {
          nodes[componentsInNetlist].bbcount = 1;
          nodes[componentsInNetlist].inputs.size += 1;
          nodes[componentsInNetlist]
              .inputs.input[nodes[componentsInNetlist].inputs.size - 1]
              .type = "c";
          nodes[componentsInNetlist]
              .inputs.input[nodes[componentsInNetlist].inputs.size - 1]
              .bitSize = 32;
          nodes[componentsInNetlist]
              .inputs.input[nodes[componentsInNetlist].inputs.size - 1]
              .infoType = "fake"; // Andrea 20200128 Try to force 0 to inputs.
          nodes[componentsInNetlist]
              .inputs.input[nodes[componentsInNetlist].inputs.size - 1]
              .port = 0; // Andrea 20200211
        }
      }
      if (parameter.find("ldcount") != std::string::npos) {
        nodes[componentsInNetlist].loadCount = getComponentBbcount(indx);
        if (nodes[componentsInNetlist].loadCount == 0) {
          nodes[componentsInNetlist].loadCount = 1;
          nodes[componentsInNetlist].inputs.size += 1;
          nodes[componentsInNetlist]
              .inputs.input[nodes[componentsInNetlist].inputs.size - 1]
              .type = "l";
          nodes[componentsInNetlist]
              .inputs.input[nodes[componentsInNetlist].inputs.size - 1]
              .infoType = "a";
          // Lana 9.6.2021 change to address bitwidth
          nodes[componentsInNetlist]
              .inputs.input[nodes[componentsInNetlist].inputs.size - 1]
              .bitSize = nodes[componentsInNetlist].addressSize;
          nodes[componentsInNetlist]
              .inputs.input[nodes[componentsInNetlist].inputs.size - 1]
              .port = 0; // Andrea 20200424
          nodes[componentsInNetlist].outputs.size += 1;
          nodes[componentsInNetlist]
              .outputs.output[nodes[componentsInNetlist].outputs.size - 1]
              .type = "l";
          // Lana 9.6.2021 change to data and data bitwidth
          nodes[componentsInNetlist]
              .outputs.output[nodes[componentsInNetlist].outputs.size - 1]
              .infoType = "d";
          nodes[componentsInNetlist]
              .outputs.output[nodes[componentsInNetlist].outputs.size - 1]
              .bitSize = nodes[componentsInNetlist].dataSize;
          nodes[componentsInNetlist]
              .outputs.output[nodes[componentsInNetlist].outputs.size - 1]
              .port = 0; // Andrea 20200424
        }
      }
      if (parameter.find("stcount") != std::string::npos) {
        nodes[componentsInNetlist].storeCount = getComponentBbcount(indx);
        if (nodes[componentsInNetlist].storeCount == 0) {
          nodes[componentsInNetlist].storeCount = 1;
          nodes[componentsInNetlist].inputs.size += 1;
          nodes[componentsInNetlist]
              .inputs.input[nodes[componentsInNetlist].inputs.size - 1]
              .type = "s";
          nodes[componentsInNetlist]
              .inputs.input[nodes[componentsInNetlist].inputs.size - 1]
              .infoType = "a";
          nodes[componentsInNetlist]
              .inputs.input[nodes[componentsInNetlist].inputs.size - 1]
              .port = 0; // Andrea 20200424
          // Lana 9.6.2021 change to address bitwidth
          nodes[componentsInNetlist]
              .inputs.input[nodes[componentsInNetlist].inputs.size - 1]
              .bitSize =
              nodes[componentsInNetlist].addressSize; // 32; //Andrea 20200424

          nodes[componentsInNetlist].inputs.size += 1;
          nodes[componentsInNetlist]
              .inputs.input[nodes[componentsInNetlist].inputs.size - 1]
              .type = "s";
          nodes[componentsInNetlist]
              .inputs.input[nodes[componentsInNetlist].inputs.size - 1]
              .infoType = "d";
          nodes[componentsInNetlist]
              .inputs.input[nodes[componentsInNetlist].inputs.size - 1]
              .port = 0; // Andrea 20200424
          // Lana 9.6.2021 change to data bitwidth
          nodes[componentsInNetlist]
              .inputs.input[nodes[componentsInNetlist].inputs.size - 1]
              .bitSize =
              nodes[componentsInNetlist].dataSize; // 32; //Andrea 20200424
        }
      }
      if (parameter.find("memAddress") != std::string::npos) {
        nodes[componentsInNetlist].memAddress = getComponentMemAddress(indx);
      }
      if (parameter.find("bbId") != std::string::npos) {
        nodes[componentsInNetlist].bbId = getComponentBbId(indx);
      }
      if (parameter.find("portId") != std::string::npos) {
        nodes[componentsInNetlist].portId = getComponentPortId(indx);
      }
      if (parameter.find("offset") != std::string::npos) {
        nodes[componentsInNetlist].offset = getComponentOffset(indx);
      }

      if (parameter.find("fifoDepth") != std::string::npos) {
        nodes[componentsInNetlist].fifodepth = getComponentBbcount(indx);
      }

      // Jiantao, 14/06/2022, Separate the depth for Load and Store Queue
      if (parameter.find("fifoDepth_L") != std::string::npos) {
        nodes[componentsInNetlist].fifodepthL = getComponentBbcount(indx);
      }

      if (parameter.find("fifoDepth_S") != std::string::npos) {
        nodes[componentsInNetlist].fifodepthS = getComponentBbcount(indx);
      }

      if (parameter.find("numLoads") != std::string::npos) {
        nodes[componentsInNetlist].numLoads = getComponentNumloads(indx);
      }
      if (parameter.find("numStores") != std::string::npos) {
        nodes[componentsInNetlist].numStores = getComponentNumstores(indx);
      }
      if (parameter.find("loadOffsets") != std::string::npos) {
        nodes[componentsInNetlist].loadOffsets = getComponentNumstores(indx);
      }
      if (parameter.find("storeOffsets") != std::string::npos) {
        nodes[componentsInNetlist].storeOffsets = getComponentNumstores(indx);
      }
      if (parameter.find("loadPorts") != std::string::npos) {
        nodes[componentsInNetlist].loadPorts = getComponentNumstores(indx);
      }
      if (parameter.find("storePorts") != std::string::npos) {
        // nodes[componentsInNetlist].storePorts = get_component_numstores(
        // parameters[indx] );
        nodes[componentsInNetlist].storePorts =
            stripExtension(getComponentNumstores(indx), "];");
      }
      if (parameter.find("orderings") != std::string::npos) {
        nodes[componentsInNetlist].orderings = getComponentOrderings(indx);
      }
      if (parameter.find("constants") != std::string::npos) {
        nodes[componentsInNetlist].constants = getComponentConstants(indx);
      }
    }
    if (nodes[componentsInNetlist].type == "Buffer" &&
        nodes[componentsInNetlist].slots == 1) {
      if (nodes[componentsInNetlist].trasparent) {
        nodes[componentsInNetlist].type = "TEHB";
      }
      nodes[componentsInNetlist].componentOperator =
          nodes[componentsInNetlist].type;
    }
    if (nodes[componentsInNetlist].type == "Buffer" &&
        nodes[componentsInNetlist].slots == 2) {
      if (nodes[componentsInNetlist].trasparent) {
        nodes[componentsInNetlist].type = "tFifo";
      }
      nodes[componentsInNetlist].componentOperator =
          nodes[componentsInNetlist].type;
    }

    if (nodes[componentsInNetlist].type == "Fifo") {
      if (nodes[componentsInNetlist].trasparent) {
        nodes[componentsInNetlist].type = "tFifo";
      } else {
        nodes[componentsInNetlist].type = "nFifo";
      }
      nodes[componentsInNetlist].componentOperator =
          nodes[componentsInNetlist].type;
    }

    componentsInNetlist++;
    if (componentsInNetlist >= MAX_NODES) {
      cerr << "The number of components in the netlist exceed the maximum "
              "allowed "
           << MAX_NODES << endl;
    }
  }
}

void parseDOT(const string &filename) {
  componentsInNetlist = 0;

  string line;
  // the input .DOT file is read twice, because parseConnections assumes that
  // the pred and succ of the connection are already parsed
  ifstream inFile(filename);
  if (!inFile.is_open()) {
    cerr << "File " << filename << " not found " << endl << endl << endl;
    exit(EXIT_FAILURE);
  }
  while (getline(inFile, line, '\n')) {
    line = checkComments(line);
    vector<string> splittedStr;
    stringSplit(line, '[', splittedStr);
    if (!splittedStr.empty() && !isConnection(line))
      parseComponents(splittedStr[0], splittedStr[1]);
  }
  inFile.close();
  inFile.open(filename);
  if (!inFile.is_open()) {
    cerr << "File " << filename << " not found " << endl << endl << endl;
    exit(EXIT_FAILURE);
  }
  while (getline(inFile, line, '\n')) {
    line = checkComments(line);
    vector<string> splittedStr;
    stringSplit(line, '[', splittedStr);
    if (!splittedStr.empty() && isConnection(line))
      parseConnections(line);
  }
  inFile.close();
}
