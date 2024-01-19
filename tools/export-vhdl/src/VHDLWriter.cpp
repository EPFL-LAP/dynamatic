//===- VHDLWriter.cpp - Generate VHDL corresponding to DOT ------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VHDLWriter.h"
#include "DOTParser.h"
#include "LSQGenerator.h"
#include "StringUtils.h"
#include <algorithm>
#include <bits/stdc++.h>
#include <cctype>
#include <fstream>
#include <iostream>
#include <list>
#include <math.h>
#include <string>
#include <vector>

using namespace std;

namespace {
struct Component {
  int inPorts;
  vector<string> inPortsNameStr;
  vector<string> inPortsTypeStr;
  int outPorts;
  vector<string> outPortsNameStr;
  vector<string> outPortsTypeStr;
};
} // namespace

static string entityName[] = {
    ENTITY_MERGE,       ENTITY_READ_MEMORY, ENTITY_SINGLE_OP,
    ENTITY_GET_PTR,     ENTITY_INT_MUL,     ENTITY_INT_ADD,
    ENTITY_INT_SUB,     ENTITY_BUF,         ENTITY_TEHB,
    ENTITY_OEHB,        ENTITY_FIFO,        ENTITY_NFIFO,
    ENTITY_TFIFO,       ENTITY_FORK,        ENTITY_LFORK,
    ENTITY_ICMP,        ENTITY_CONSTANT,    ENTITY_BRANCH,
    ENTITY_END,         ENTITY_START,       ENTITY_WRITE_MEMORY,
    ENTITY_SOURCE,      ENTITY_SINK,        ENTITY_CTRLMERGE,
    ENTITY_MUX,         ENTITY_LSQ,         ENTITY_MC,
    ENTITY_DISTRIBUTOR, ENTITY_INJECTOR,    ENTITY_SELECTOR};

static string componentTypes[] = {
    COMPONENT_MERGE,       COMPONENT_READ_MEMORY, COMPONENT_SINGLE_OP,
    COMPONENT_GET_PTR,     COMPONENT_INT_MUL,     COMPONENT_INT_ADD,
    COMPONENT_INT_SUB,     COMPONENT_BUF,         COMPONENT_TEHB,
    COMPONENT_OEHB,        COMPONENT_FIFO,        COMPONENT_NFIFO,
    COMPONENT_TFIFO,       COMPONENT_FORK,        COMPONENT_LFORK,
    COMPONENT_ICMP,        COMPONENT_CONSTANT_,   COMPONENT_BRANCH,
    COMPONENT_END,         COMPONENT_START,       COMPONENT_WRITE_MEMORY,
    COMPONENT_SOURCE,      COMPONENT_SINK,        COMPONENT_CTRLMERGE,
    COMPONENT_MUX,         COMPONENT_LSQ,         COMPONENT_MC,
    COMPONENT_DISTRIBUTOR, COMPONENT_INJECTOR,    COMPONENT_SELECTOR};

static string inputsName[] = {DATAIN_ARRAY, PVALID_ARRAY, NREADY_ARRAY

};

static string inputsType[] = {"std_logic_vector(", "std_logic",
                              "std_logic_vector("

};

static string outputsName[] = {DATAOUT_ARRAY, READY_ARRAY, VALID_ARRAY

};

static string outputsType[] = {"std_logic_vector(", "std_logic_vector(",
                               "std_logic_vector("

};

static vector<string> inPortsNameGeneric(inputsName,
                                         inputsName + sizeof(inputsName) /
                                                          sizeof(string));
static vector<string> inPortsTypeGeneric(inputsType,
                                         inputsType + sizeof(inputsType) /
                                                          sizeof(string));
static vector<string> outPortsNameGeneric(outputsName,
                                          outputsName + sizeof(outputsName) /
                                                            sizeof(string));
static vector<string> outPortsTypeGeneric(outputsType,
                                          outputsType + sizeof(outputsType) /
                                                            sizeof(string));

Component componentsType[MAX_COMPONENTS];

ofstream netlist;
ofstream tbWrapper;

static void writeSignals() {
  int indx;
  string signal;

  for (int i = 0; i < componentsInNetlist; i++) {
    if ((nodes[i].name.empty())) // Check if the name is not empty
    {
      cout << "**Warning: node " << i
           << " does not have an instance name -- skipping node **" << endl;
    } else {
      netlist << endl;
      netlist << "\t" << SIGNAL_STRING << nodes[i].name << "_clk : std_logic;"
              << endl;
      netlist << "\t" << SIGNAL_STRING << nodes[i].name << "_rst : std_logic;"
              << endl;

      for (indx = 0; indx < nodes[i].inputs.size; indx++) {

        //                     if ( nodes[i].type == "Branch" && indx == 1 )
        //                     {
        //                             signal = SIGNAL;
        //                             signal += nodes[i].name;
        //                             signal += UNDERSCORE;
        //                             signal += "Condition_valid";
        //
        //                             signal += UNDERSCORE;
        //                             signal += to_string( indx );
        //                             signal += COLOUMN;
        //                             signal += " std_logic;";
        //                             signal += '\n';
        //
        //                             netlist << "\t"  << signal ;
        //
        //                     }
        //                     else
        {
          // for ( int in_port_indx = 0; in_port_indx <
          // components_type[nodes[i].componentType].in_ports; in_port_indx++ )
          for (int inPortIndx = 0; inPortIndx < 1; inPortIndx++) {
            signal = SIGNAL_STRING;
            signal += nodes[i].name;
            signal += UNDERSCORE;
            signal += componentsType[0].inPortsNameStr[inPortIndx];

            signal += UNDERSCORE;
            signal += to_string(indx);
            signal += COLOUMN;
            // Lana 20.01.22 Branch condition no longer needs to be treated as a
            // special case Dot specifies branch condition bitwidth correctly,
            // so just rea value like for any other port
            // if ( nodes[i].type == "Branch" && indx == 1 )
            //{
            // signal +="std_logic_vector (0 downto 0);";
            //}
            // else
            if (nodes[i].type == COMPONENT_DISTRIBUTOR && indx == 1) {
              int condSize =
                  nodes[i].inputs.input[nodes[i].inputs.size - 1].bitSize;
              signal += "std_logic_vector (" + to_string(condSize - 1) +
                        " downto 0);";
            } else {
              signal += componentsType[0].inPortsTypeStr[inPortIndx];
              signal += to_string((nodes[i].inputs.input[indx].bitSize - 1 >= 0)
                                      ? nodes[i].inputs.input[indx].bitSize - 1
                                      : DEFAULT_BITWIDTH - 1);
              signal += " downto 0);";
            }
            signal += '\n';
            netlist << "\t" << signal;
          }
        }
      }
      for (indx = 0; indx < nodes[i].inputs.size; indx++) {

        // Write the Valid Signals
        signal = SIGNAL_STRING;
        signal += nodes[i].name;
        signal += UNDERSCORE;
        signal += PVALID_ARRAY; // Valid
        signal += UNDERSCORE;
        signal += to_string(indx);
        signal += COLOUMN;
        signal += STD_LOGIC;
        signal += '\n';
        netlist << "\t" << signal;
      }
      for (indx = 0; indx < nodes[i].inputs.size; indx++) {

        // Write the Ready Signals
        signal = SIGNAL_STRING;
        signal += nodes[i].name;
        signal += UNDERSCORE;
        signal += READY_ARRAY; // Valid
        signal += UNDERSCORE;
        signal += to_string(indx);
        signal += COLOUMN;
        signal += STD_LOGIC;
        signal += '\n';
        netlist << "\t" << signal;
      }

      for (indx = 0; indx < nodes[i].outputs.size; indx++) {
        // Write the Ready Signals
        signal = SIGNAL_STRING;
        signal += nodes[i].name;
        signal += UNDERSCORE;
        signal += NREADY_ARRAY; // Ready
        signal += UNDERSCORE;
        signal += to_string(indx);
        signal += COLOUMN;
        signal += STD_LOGIC;
        signal += '\n';
        netlist << "\t" << signal;

        // Write the Valid Signals
        signal = SIGNAL_STRING;
        signal += nodes[i].name;
        signal += UNDERSCORE;
        signal += VALID_ARRAY; // Ready
        signal += UNDERSCORE;
        signal += to_string(indx);
        signal += COLOUMN;
        signal += STD_LOGIC;
        signal += '\n';
        netlist << "\t" << signal;

        for (int outPortIndx = 0;
             outPortIndx < componentsType[nodes[i].componentType].outPorts;
             outPortIndx++) {

          signal = SIGNAL_STRING;
          signal += nodes[i].name;
          signal += UNDERSCORE;
          signal += componentsType[0].outPortsNameStr[outPortIndx];
          signal += UNDERSCORE;
          signal += to_string(indx);
          signal += COLOUMN;
          signal += componentsType[0].outPortsTypeStr[outPortIndx];
          signal += to_string((nodes[i].outputs.output[indx].bitSize - 1 >= 0)
                                  ? nodes[i].outputs.output[indx].bitSize - 1
                                  : DEFAULT_BITWIDTH - 1);
          signal += " downto 0);";
          signal += '\n';

          netlist << "\t" << signal;
        }
      }
    }

    if (nodes[i].type == "Exit") {

      signal = SIGNAL_STRING;
      signal += nodes[i].name;
      signal += UNDERSCORE;
      // signal += "validArray_0";
      signal += "validArray";
      signal += UNDERSCORE;
      signal += to_string(indx);
      signal += COLOUMN;
      signal += " std_logic;";
      signal += '\n';

      netlist << "\t" << signal;

      signal = SIGNAL_STRING;
      signal += nodes[i].name;
      signal += UNDERSCORE;
      // signal += "dataOutArray_0";
      signal += "dataOutArray";
      signal += UNDERSCORE;
      signal += to_string(indx);
      signal += COLOUMN;
      signal += " std_logic_vector (31 downto 0);";
      signal += '\n';

      netlist << "\t" << signal;

      signal = SIGNAL_STRING;
      signal += nodes[i].name;
      signal += UNDERSCORE;
      // signal += "nReadyArray_0";
      signal += "nReadyArray";
      signal += UNDERSCORE;
      signal += to_string(indx);
      signal += COLOUMN;
      signal += " std_logic;";
      signal += '\n';

      netlist << "\t" << signal;
    }

    if (nodes[i].type == "LSQ") {
      signal = SIGNAL_STRING;
      signal += nodes[i].name;
      signal += UNDERSCORE;
      signal += "io_queueEmpty";
      signal += COLOUMN;
      signal += STD_LOGIC;
      netlist << "\t" << signal << endl;
    }

    if (nodes[i].type == "MC" || nodes[i].type == "LSQ") {
      signal = SIGNAL_STRING;
      signal += nodes[i].name;
      signal += UNDERSCORE;
      signal += "we0_ce0";
      signal += COLOUMN;
      signal += STD_LOGIC;
      netlist << "\t" << signal << endl;
    }

    // LSQ-MC Modifications
    if (nodes[i].type.find("LSQ") != std::string::npos) {

      netlist << "\t" << SIGNAL_STRING << nodes[i].name
              << "_address0 : std_logic_vector (" << (nodes[i].addressSize - 1)
              << " downto 0);" << endl;
      netlist << "\t" << SIGNAL_STRING << nodes[i].name << "_ce0 : std_logic;"
              << endl;
      netlist << "\t" << SIGNAL_STRING << nodes[i].name << "_we0 : std_logic;"
              << endl;
      netlist << "\t" << SIGNAL_STRING << nodes[i].name
              << "_dout0 : std_logic_vector (31 downto 0);" << endl;
      netlist << "\t" << SIGNAL_STRING << nodes[i].name
              << "_din0 : std_logic_vector (31 downto 0);" << endl;

      netlist << "\t" << SIGNAL_STRING << nodes[i].name
              << "_address1 : std_logic_vector (" << (nodes[i].addressSize - 1)
              << " downto 0);" << endl;
      netlist << "\t" << SIGNAL_STRING << nodes[i].name << "_ce1 : std_logic;"
              << endl;
      netlist << "\t" << SIGNAL_STRING << nodes[i].name << "_we1 : std_logic;"
              << endl;
      netlist << "\t" << SIGNAL_STRING << nodes[i].name
              << "_dout1 : std_logic_vector (31 downto 0);" << endl;
      netlist << "\t" << SIGNAL_STRING << nodes[i].name
              << "_din1 : std_logic_vector (31 downto 0);" << endl;

      netlist << "\t" << SIGNAL_STRING << nodes[i].name
              << "_load_ready : std_logic;" << endl;
      netlist << "\t" << SIGNAL_STRING << nodes[i].name
              << "_store_ready : std_logic;" << endl;
    }
  }
}

static void writeConnections() {
  string signal1, signal2;

  netlist << endl;

  for (int i = 0; i < componentsInNetlist; i++) {
    netlist << endl;

    netlist << "\t" << nodes[i].name << UNDERSCORE << "clk"
            << " <= "
            << "clk" << SEMICOLOUMN << endl;
    netlist << "\t" << nodes[i].name << UNDERSCORE << "rst"
            << " <= "
            << "rst" << SEMICOLOUMN << endl;

    if (nodes[i].type == "MC") {
      signal1 = nodes[i].memory;
      signal1 += UNDERSCORE;
      signal1 += "ce0";

      signal2 = nodes[i].name;
      signal2 += UNDERSCORE;
      signal2 += "we0_ce0";

      netlist << "\t" << signal1 << " <= " << signal2 << SEMICOLOUMN << endl;

      signal1 = nodes[i].memory;
      signal1 += UNDERSCORE;
      signal1 += "we0";

      signal2 = nodes[i].name;
      signal2 += UNDERSCORE;
      signal2 += "we0_ce0";

      netlist << "\t" << signal1 << " <= " << signal2 << SEMICOLOUMN << endl;
    }

    // LSQ-MC Modifications
    if (nodes[i].type == "LSQ") {
      bool mcLsq = false;

      for (int indx = 0; indx < nodes[i].inputs.size; indx++) {
        // cout << nodes[i].name << " input " << indx << " type " <<
        // nodes[i].inputs.input[indx].type << endl;
        if (nodes[i].inputs.input[indx].type == "x") {
          // if x port exists, lsq is connected to mc and not to memory
          // directly
          mcLsq = true;

          // - for the port x0d:
          // LSQ_x_din1 <= LSQ_x_dataInArray_4;
          // LSQ_x_readyArray_4 <= '1';

          signal1 = nodes[i].name;
          signal1 += UNDERSCORE;
          signal1 += "din1";

          signal2 = nodes[i].name;
          signal2 += UNDERSCORE;
          signal2 += DATAIN_ARRAY;
          signal2 += UNDERSCORE;
          signal2 += to_string(indx);

          netlist << "\t" << signal1 << " <= " << signal2 << SEMICOLOUMN
                  << endl;

          signal1 = nodes[i].name;
          signal1 += UNDERSCORE;
          signal1 += READY_ARRAY;
          signal1 += UNDERSCORE;
          signal1 += to_string(indx);

          signal2 = "'1'";

          netlist << "\t" << signal1 << " <= " << signal2 << SEMICOLOUMN
                  << endl;
        }
        // if ( nodes[i].inputs.input[indx].type == "y" )
        {}
      }

      if (!mcLsq) {
        signal1 = nodes[i].name;
        signal1 += UNDERSCORE;
        signal1 += "din1";

        signal2 = nodes[i].memory;
        signal2 += UNDERSCORE;
        signal2 += "din1";

        netlist << "\t" << signal1 << " <= " << signal2 << SEMICOLOUMN << endl;

        signal1 = nodes[i].name;
        signal1 += UNDERSCORE;
        signal1 += "store_ready";

        signal2 = "'1'";

        netlist << "\t" << signal1 << " <= " << signal2 << SEMICOLOUMN << endl;

        signal1 = nodes[i].name;
        signal1 += UNDERSCORE;
        signal1 += "load_ready";

        signal2 = "'1'";

        netlist << "\t" << signal1 << " <= " << signal2 << SEMICOLOUMN << endl;
      }

      for (int indx = 0; indx < nodes[i].outputs.size; indx++) {
        // cout << nodes[i].name << " output " << indx << " type " <<
        // nodes[i].outputs.output[indx].type << endl;

        if (nodes[i].outputs.output[indx].type == "x") {
          //- for the port x0a, check the index (in this case, it's out3) and
          // build a load address interface as follows:
          // LSQ_x_load_ready <= LSQ_x_nReadyArray_2;
          // LSQ_x_dataOutArray_2 <= LSQ_x_address1;
          // LSQ_x_validArray_2 <= LSQ_x_ce1;

          signal1 = nodes[i].name;
          signal1 += UNDERSCORE;
          signal1 += "load_ready";

          signal2 = nodes[i].name;
          signal2 += UNDERSCORE;
          signal2 += NREADY_ARRAY;
          signal2 += UNDERSCORE;
          signal2 += to_string(indx);

          netlist << "\t" << signal1 << " <= " << signal2 << SEMICOLOUMN
                  << endl;

          signal1 = nodes[i].name;
          signal1 += UNDERSCORE;
          signal1 += DATAOUT_ARRAY;
          signal1 += UNDERSCORE;
          signal1 += to_string(indx);

          signal2 = nodes[i].name;
          signal2 += UNDERSCORE;
          signal2 += "address1";

          netlist << "\t" << signal1 << " <= " << signal2 << SEMICOLOUMN
                  << endl;

          signal1 = nodes[i].name;
          signal1 += UNDERSCORE;
          signal1 += VALID_ARRAY;
          signal1 += UNDERSCORE;
          signal1 += to_string(indx);

          signal2 = nodes[i].name;
          signal2 += UNDERSCORE;
          signal2 += "ce1";

          netlist << "\t" << signal1 << " <= " << signal2 << SEMICOLOUMN
                  << endl;

        } else if (nodes[i].outputs.output[indx].type == "y") {

          if (nodes[i].outputs.output[indx].infoType == "a") {
            //
            // - for the port y0a:
            // LSQ_x_validArray_3 <= LSQ_x_we0_ce0;
            // LSQ_x_store_ready <= LSQ_x_nReadyArray_3;
            // LSQ_x_dataOutArray_3 <= LSQ_x_address0;
            signal1 = nodes[i].name;
            signal1 += UNDERSCORE;
            signal1 += VALID_ARRAY;
            signal1 += UNDERSCORE;
            signal1 += to_string(indx);

            signal2 = nodes[i].name;
            signal2 += UNDERSCORE;
            signal2 += "we0_ce0";

            netlist << "\t" << signal1 << " <= " << signal2 << SEMICOLOUMN
                    << endl;

            signal1 = nodes[i].name;
            signal1 += UNDERSCORE;
            signal1 += "store_ready";

            signal2 = nodes[i].name;
            signal2 += UNDERSCORE;
            signal2 += NREADY_ARRAY;
            signal2 += UNDERSCORE;
            signal2 += to_string(indx);

            netlist << "\t" << signal1 << " <= " << signal2 << SEMICOLOUMN
                    << endl;

            signal1 = nodes[i].name;
            signal1 += UNDERSCORE;
            signal1 += DATAOUT_ARRAY;
            signal1 += UNDERSCORE;
            signal1 += to_string(indx);

            signal2 = nodes[i].name;
            signal2 += UNDERSCORE;
            signal2 += "address0";

            netlist << "\t" << signal1 << " <= " << signal2 << SEMICOLOUMN
                    << endl;

          } else if (nodes[i].outputs.output[indx].infoType == "d") {
            //
            // - for the port y0d:
            // LSQ_x_validArray_4 <= LSQ_x_we0_ce0;
            // LSQ_x_dataOutArray_4 <= LSQ_x_dout0;
            signal1 = nodes[i].name;
            signal1 += UNDERSCORE;
            signal1 += VALID_ARRAY;
            signal1 += UNDERSCORE;
            signal1 += to_string(indx);

            signal2 = nodes[i].name;
            signal2 += UNDERSCORE;
            signal2 += "we0_ce0";

            netlist << "\t" << signal1 << " <= " << signal2 << SEMICOLOUMN
                    << endl;

            signal1 = nodes[i].name;
            signal1 += UNDERSCORE;
            signal1 += DATAOUT_ARRAY;
            signal1 += UNDERSCORE;
            signal1 += to_string(indx);

            signal2 = nodes[i].name;
            signal2 += UNDERSCORE;
            signal2 += "dout0";

            netlist << "\t" << signal1 << " <= " << signal2 << SEMICOLOUMN
                    << endl;
          }
        }
      }

      if (!mcLsq) {

        signal1 = nodes[i].memory;
        signal1 += UNDERSCORE;
        signal1 += "address1";

        signal2 = nodes[i].name;
        signal2 += UNDERSCORE;
        signal2 += "address1";

        // netlist << "\t"  << signal_1  << " <= " << signal_2 << SEMICOLOUMN
        // << endl;

        netlist << "\t" << signal1 << " <= std_logic_vector (resize(unsigned("
                << signal2 << ")," << signal1 << "'length))" << SEMICOLOUMN
                << endl;

        signal1 = nodes[i].memory;
        signal1 += UNDERSCORE;
        signal1 += "ce1";

        signal2 = nodes[i].name;
        signal2 += UNDERSCORE;
        signal2 += "ce1";

        netlist << "\t" << signal1 << " <= " << signal2 << SEMICOLOUMN << endl;

        signal1 = nodes[i].memory;
        signal1 += UNDERSCORE;
        signal1 += "address0";

        signal2 = nodes[i].name;
        signal2 += UNDERSCORE;
        signal2 += "address0";

        // netlist << "\t"  << signal_1  << " <= " << signal_2 << SEMICOLOUMN
        // << endl;
        netlist << "\t" << signal1 << " <= std_logic_vector (resize(unsigned("
                << signal2 << ")," << signal1 << "'length))" << SEMICOLOUMN
                << endl;

        signal1 = nodes[i].memory;
        signal1 += UNDERSCORE;
        signal1 += "ce0";

        signal2 = nodes[i].name;
        signal2 += UNDERSCORE;
        signal2 += "we0_ce0";

        netlist << "\t" << signal1 << " <= " << signal2 << SEMICOLOUMN << endl;

        signal1 = nodes[i].memory;
        signal1 += UNDERSCORE;
        signal1 += "we0";

        signal2 = nodes[i].name;
        signal2 += UNDERSCORE;
        signal2 += "we0_ce0";

        netlist << "\t" << signal1 << " <= " << signal2 << SEMICOLOUMN << endl;

        signal1 = nodes[i].memory;
        signal1 += UNDERSCORE;
        signal1 += "dout0";

        signal2 = nodes[i].name;
        signal2 += UNDERSCORE;
        signal2 += "dout0";

        netlist << "\t" << signal1 << " <= " << signal2 << SEMICOLOUMN << endl;
      }
    }

    if (nodes[i].type == "Entry") {

      if (!(nodes[i].name.find("start") != std::string::npos)) // If not start
        if (!(nodes[i].componentControl)) {
          signal1 = nodes[i].name;
          signal1 += UNDERSCORE;
          signal1 += DATAIN_ARRAY;
          signal1 += UNDERSCORE;
          signal1 += "0";

          signal2 = nodes[i].name;
          signal2 += UNDERSCORE;
          signal2 += "din";

          netlist << "\t" << signal1 << " <= " << signal2 << SEMICOLOUMN
                  << endl;
        }

      signal1 = nodes[i].name;
      signal1 += UNDERSCORE;
      signal1 += PVALID_ARRAY;
      signal1 += UNDERSCORE;
      signal1 += "0";

      // signal_2 = "ap_start";

      //                 signal_2 = nodes[i].name;
      //                 signal_2 += UNDERSCORE;
      //                 signal_2 +="valid_in";

      signal2 = "start_valid";

      netlist << "\t" << signal1 << " <= " << signal2 << SEMICOLOUMN << endl;

      if ((nodes[i].name.find("start") != std::string::npos)) {
        signal1 = nodes[i].name;
        signal1 += UNDERSCORE;
        signal1 += READY_ARRAY;
        signal1 += UNDERSCORE;
        signal1 += "0";

        // signal_2 = "ap_start";

        //                 signal_2 = nodes[i].name;
        //                 signal_2 += UNDERSCORE;
        //                 signal_2 +="valid_in";

        signal2 = "start_ready";

        netlist << "\t" << signal2 << " <= " << signal1 << SEMICOLOUMN << endl;
      }
    }

    if (nodes[i].type == "Exit") {

      signal1 = "end_valid";

      signal2 = nodes[i].name;
      signal2 += UNDERSCORE;
      signal2 += VALID_ARRAY;

      netlist << "\t" << signal1 << " <= " << signal2 << UNDERSCORE << "0"
              << SEMICOLOUMN << endl;

      signal1 = "end_out";

      signal2 = nodes[i].name;
      signal2 += UNDERSCORE;
      signal2 += DATAOUT_ARRAY;

      netlist << "\t" << signal1 << " <= " << signal2 << UNDERSCORE << "0"
              << SEMICOLOUMN << endl;

      signal1 = nodes[i].name;
      signal1 += UNDERSCORE;
      signal1 += NREADY_ARRAY;
      signal1 += UNDERSCORE;
      signal1 += "0";

      signal2 = "end_ready";

      netlist << "\t" << signal1 << " <= " << signal2 << SEMICOLOUMN << endl;
    }

    for (int indx = 0; indx < nodes[i].outputs.size; indx++) {
      if (nodes[i].outputs.output[indx].nextNodesID != COMPONENT_NOT_FOUND) {
        signal1 = nodes[nodes[i].outputs.output[indx].nextNodesID].name;
        signal1 += UNDERSCORE;
        signal1 += PVALID_ARRAY;
        signal1 += UNDERSCORE;
        signal1 += to_string(nodes[i].outputs.output[indx].nextNodesPort);

        signal2 = nodes[i].name;
        signal2 += UNDERSCORE;
        signal2 += VALID_ARRAY;
        signal2 += UNDERSCORE;
        signal2 += to_string(indx);

        netlist << "\t" << signal1 << " <= " << signal2 << SEMICOLOUMN << endl;
      }

      if (nodes[i].outputs.output[indx].nextNodesID != COMPONENT_NOT_FOUND) {

        signal1 = nodes[i].name;
        signal1 += UNDERSCORE;
        signal1 += NREADY_ARRAY;
        signal1 += UNDERSCORE;
        signal1 += to_string(indx);

        signal2 = nodes[nodes[i].outputs.output[indx].nextNodesID].name;
        // signal_2 = (nodes[i].outputs.output[indx].nextNodesID ==
        // COMPONENT_NOT_FOUND ) ? "unconnected" :
        // nodes[nodes[i].outputs.output[indx].nextNodesID].name;
        signal2 += UNDERSCORE;
        signal2 += READY_ARRAY;
        signal2 += UNDERSCORE;
        signal2 += to_string(nodes[i].outputs.output[indx].nextNodesPort);

        // outFile << "\t"  << signal_1 <<
        // nodes[i].outputs.output[indx].nextNodesPort << " <= " << signal_2
        // << UNDERSCORE << indx <<SEMICOLOUMN << endl;
        netlist << "\t" << signal1 << " <= " << signal2 << SEMICOLOUMN << endl;
      }

      // for ( int in_port_indx = 0; in_port_indx <
      // components_type[nodes[i].componentType].in_ports; in_port_indx++)
      for (int inPortIndx = 0; inPortIndx < 1; inPortIndx++) {

        if (nodes[i].outputs.output[indx].nextNodesID != COMPONENT_NOT_FOUND) {

          signal1 = nodes[nodes[i].outputs.output[indx].nextNodesID].name;
          // signal_1 = (nodes[i].outputs.output[indx].nextNodesID ==
          // COMPONENT_NOT_FOUND ) ? "unconnected" :
          // nodes[nodes[i].outputs.output[indx].nextNodesID].name;
          signal1 += UNDERSCORE;
          signal1 += componentsType[0].inPortsNameStr[inPortIndx];
          signal1 += UNDERSCORE;
          signal1 += to_string(nodes[i].outputs.output[indx].nextNodesPort);

          // inverted
          string inverted;

          if (nodes[nodes[i].outputs.output[indx].nextNodesID]
                  .inputs.input[nodes[i].outputs.output[indx].nextNodesPort]
                  .type == "i") {
            inverted = "not "; // inverted
          } else {
            inverted = "";
          }

          signal2 = nodes[i].name;
          signal2 += UNDERSCORE;
          signal2 += componentsType[0].outPortsNameStr[inPortIndx];
          signal2 += UNDERSCORE;
          signal2 += to_string(indx);

          if (nodes[nodes[i].outputs.output[indx].nextNodesID].type.find(
                  "Constant") !=
              std::string::npos) // Overwrite predecessor with constant value
          {
            signal2 = "\"";
            signal2 += stringConstant(
                nodes[nodes[i].outputs.output[indx].nextNodesID].componentValue,
                nodes[nodes[i].outputs.output[indx].nextNodesID]
                    .inputs.input[0]
                    .bitSize);
            signal2 += "\"";
            netlist << "\t" << signal1 << " <= " << signal2 << SEMICOLOUMN
                    << endl;
          } else {
            netlist << "\t" << signal1 << " <= " << inverted
                    << "std_logic_vector (resize(unsigned(" << signal2 << "),"
                    << signal1 << "'length))" << SEMICOLOUMN << endl;
          }
        }
      }
    }
  }
}

static string getComponentEntity(const string &component, int componentId) {
  string componentEntity;

  for (int indx = 0; indx < ENTITY_MAX; indx++) {
    // cout  << "component_id" << component_id << "component "<< component << "
    // " << component_types[indx] << endl;
    if (component.compare(componentTypes[indx]) == 0) {
      componentEntity = entityName[indx];
      break;
    }
  }
  return componentEntity;
}

static int getMemoryInputs(int nodeId) {
  int memoryInputs = nodes[nodeId].inputs.size;
  for (int indx = 0; indx < nodes[nodeId].inputs.size; indx++) {
    if (nodes[nodeId].inputs.input[indx].type != "e")
      memoryInputs--;
  }
  return memoryInputs;
}

static string getGeneric(int nodeId) {
  string generic;

  if (nodes[nodeId].inputs.input[0].bitSize == 0) {
    nodes[nodeId].inputs.input[0].bitSize = 32;
  }

  if (nodes[nodeId].outputs.output[0].bitSize == 0) {
    nodes[nodeId].outputs.output[0].bitSize = 32;
  }

  if (nodes[nodeId].type.find("Branch") != std::string::npos) {
    generic = to_string(nodes[nodeId].inputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].inputs.input[0].bitSize);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.output[0].bitSize);
  }
  if (nodes[nodeId].type.find(COMPONENT_DISTRIBUTOR) != std::string::npos) {
    // INPUTS
    generic = to_string(nodes[nodeId].inputs.size);
    generic += COMMA;
    // OUTPUTS
    generic += to_string(nodes[nodeId].outputs.size);
    generic += COMMA;
    // COND_SIZE
    generic += to_string((int)ceil(log2(nodes[nodeId].outputs.size)));
    generic += COMMA;
    // DATA_SIZE_IN
    generic += to_string(nodes[nodeId].inputs.input[0].bitSize);
    generic += COMMA;
    // DATA_SIZE_OUT
    generic += to_string(nodes[nodeId].outputs.output[0].bitSize);
  }
  if (nodes[nodeId].type.find("Buf") != std::string::npos) {
    generic = to_string(nodes[nodeId].inputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].inputs.input[0].bitSize);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.output[0].bitSize);
  }

  if (nodes[nodeId].type.find("Merge") != std::string::npos) {
    generic = to_string(nodes[nodeId].inputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].inputs.input[0].bitSize);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.output[0].bitSize);
  }

  if (nodes[nodeId].type.find("Fork") != std::string::npos) {
    generic = to_string(nodes[nodeId].inputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].inputs.input[0].bitSize);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.output[0].bitSize);
  }

  if (nodes[nodeId].type.find("LazyFork") != std::string::npos) {
    generic = to_string(nodes[nodeId].inputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].inputs.input[0].bitSize);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.output[0].bitSize);
  }

  if (nodes[nodeId].type.find("Constant") != std::string::npos) {
    generic = to_string(nodes[nodeId].inputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].inputs.input[0].bitSize);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.output[0].bitSize);
  }

  if (nodes[nodeId].type.find("Operator") != std::string::npos) {
    generic = to_string(nodes[nodeId].inputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.size);
    generic += COMMA;
    if (nodes[nodeId].componentOperator.find("select") != std::string::npos ||
        nodes[nodeId].componentOperator.find("mc_load_op") !=
            std::string::npos ||
        nodes[nodeId].componentOperator.find("mc_store_op") !=
            std::string::npos ||
        nodes[nodeId].componentOperator.find("lsq_load_op") !=
            std::string::npos ||
        nodes[nodeId].componentOperator.find("lsq_store_op") !=
            std::string::npos) {
      generic += to_string(nodes[nodeId].inputs.input[1].bitSize);
    } else {
      generic += to_string(nodes[nodeId].inputs.input[0].bitSize);
    }

    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.output[0].bitSize);
    if (nodes[nodeId].componentOperator.find("getelementptr_op") !=
        std::string::npos) {
      generic += COMMA;
      generic += to_string(nodes[nodeId].constants);
    }
  }

  if (nodes[nodeId].type.find("Entry") != std::string::npos) {
    generic = to_string(nodes[nodeId].inputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].inputs.input[0].bitSize);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.output[0].bitSize);
  }

  if (nodes[nodeId].type.find("Exit") != std::string::npos) {
    generic = to_string(nodes[nodeId].inputs.size - getMemoryInputs(nodeId));
    generic += COMMA;
    generic += to_string(getMemoryInputs(nodeId));
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.size);
    generic += COMMA;

#if 0
        int size_max = 0;
        for ( int indx = 0; indx < nodes[node_id].inputs.size; indx++ )
        {
            if ( nodes[node_id].inputs.input[indx].bitSize > size_max )
            {
                size_max = nodes[node_id].inputs.input[indx].bitSize;
            }
        }
        generic += to_string(size_max);
#endif

    // generic +=
    // to_string(nodes[node_id].inputs.input[nodes[node_id].inputs.size].bitSize);
    generic += to_string(
        nodes[nodeId].inputs.input[nodes[nodeId].inputs.size - 1].bitSize);

    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.output[0].bitSize);
  }

  if (nodes[nodeId].type.find("Sink") != std::string::npos) {
    generic = to_string(nodes[nodeId].inputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].inputs.input[0].bitSize);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.output[0].bitSize);
  }

  if (nodes[nodeId].type.find("Source") != std::string::npos) {
    generic = to_string(nodes[nodeId].inputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].inputs.input[0].bitSize);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.output[0].bitSize);
  }

  if (nodes[nodeId].type.find("Fifo") != std::string::npos) {
    generic = to_string(nodes[nodeId].inputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].inputs.input[0].bitSize);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.output[0].bitSize);
    generic += COMMA;
    generic += to_string(nodes[nodeId].slots);
  }

  if (nodes[nodeId].type.find("nFifo") != std::string::npos) {
    generic = to_string(nodes[nodeId].inputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].inputs.input[0].bitSize);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.output[0].bitSize);
    generic += COMMA;
    generic += to_string(nodes[nodeId].slots);
  }

  if (nodes[nodeId].type.find("tFifo") != std::string::npos) {
    generic = to_string(nodes[nodeId].inputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].inputs.input[0].bitSize);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.output[0].bitSize);
    generic += COMMA;
    generic += to_string(nodes[nodeId].slots);
  }

  if (nodes[nodeId].type.find("TEHB") != std::string::npos) {
    generic = to_string(nodes[nodeId].inputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].inputs.input[0].bitSize);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.output[0].bitSize);
  }
  if (nodes[nodeId].type.find("OEHB") != std::string::npos) {
    generic = to_string(nodes[nodeId].inputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].inputs.input[0].bitSize);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.output[0].bitSize);
  }

  if (nodes[nodeId].type.find("Mux") != std::string::npos) {
    generic = to_string(nodes[nodeId].inputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].inputs.input[1].bitSize);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.output[0].bitSize);
    generic += COMMA;
    generic += to_string(
        nodes[nodeId].inputs.input[0].bitSize); // condition size
                                                // inputs.input[input_indx].type
                                                /*
                                                 *
                                                 * generic += COMMA;
                                            
                                                if ( nodes[node_id].inputs.input[1].type == "i" )
                                                    generic += "1"; // input is inverted
                                                else
                                                    generic += "0"; // input is not inverted
                                                generic += COMMA;
                                                if ( nodes[node_id].inputs.input[2].type == "i" )
                                                    generic += "1"; // input is inverted
                                                else
                                                    generic += "0"; // input is not inverted
                                                */
  }

  if (nodes[nodeId].type.find("Inj") != std::string::npos) {
    generic = to_string(nodes[nodeId].inputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].inputs.input[1].bitSize);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.output[0].bitSize);
    generic += COMMA;
    generic += to_string(
        nodes[nodeId].inputs.input[0].bitSize); // condition size
                                                // inputs.input[input_indx].type
                                                /*
                                                 *
                                                 * generic += COMMA;
                                            
                                                if ( nodes[node_id].inputs.input[1].type == "i" )
                                                    generic += "1"; // input is inverted
                                                else
                                                    generic += "0"; // input is not inverted
                                                generic += COMMA;
                                                if ( nodes[node_id].inputs.input[2].type == "i" )
                                                    generic += "1"; // input is inverted
                                                else
                                                    generic += "0"; // input is not inverted
                                                */
  }

  if (nodes[nodeId].type.find("CntrlMerge") != std::string::npos) {
    generic = to_string(nodes[nodeId].inputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.size);
    generic += COMMA;
    generic += to_string(nodes[nodeId].inputs.input[0].bitSize);
    generic += COMMA;
    generic += to_string(nodes[nodeId].outputs.output[0].bitSize);
    generic += COMMA;
    generic +=
        to_string(nodes[nodeId].outputs.output[1].bitSize); // condition size
  }

  if (nodes[nodeId].type.find("MC") != std::string::npos) {
    generic += to_string(nodes[nodeId].dataSize);
    generic += COMMA;
    generic += to_string(nodes[nodeId].addressSize);
    generic += COMMA;
    generic += to_string(nodes[nodeId].bbcount);
    generic += COMMA;
    generic += to_string(nodes[nodeId].loadCount);
    generic += COMMA;
    generic += to_string(nodes[nodeId].storeCount);
  }

  if (nodes[nodeId].type.find("Selector") != std::string::npos) {

    // INPUTS : integer
    // OUTPUTS : integer
    // COND_SIZE : integer
    // DATA_SIZE_IN: integer
    // DATA_SIZE_OUT: integer
    //
    // AMOUNT_OF_BB_IDS: integer
    // AMOUNT_OF_SHARED_COMPONENTS: integer
    // BB_ID_INFO_SIZE : integer
    // BB_COUNT_INFO_SIZE : integer

    int amountOfBbs = nodes[nodeId].orderings.size();
    int bbIdInfoSize = amountOfBbs <= 1 ? 1 : (int)ceil(log2(amountOfBbs));
    int maxSharedComponents = -1;
    for (const auto &orderingPerBb : nodes[nodeId].orderings) {
      int size = orderingPerBb.size();
      if (maxSharedComponents < size) {
        maxSharedComponents = size;
      }
    }
    int bbCountInfoSize =
        maxSharedComponents <= 1 ? 1 : ceil(log2(maxSharedComponents));

    // INPUTS
    generic += to_string(nodes[nodeId].inputs.size - amountOfBbs);
    generic += COMMA;
    // OUTPUTS
    generic += to_string(nodes[nodeId].outputs.size);
    generic += COMMA;
    // COND_SIZE
    generic += to_string(
        nodes[nodeId].outputs.output[nodes[nodeId].outputs.size - 1].bitSize);
    generic += COMMA;
    // DATA_SIZE_IN
    generic += to_string(nodes[nodeId].inputs.input[0].bitSize);
    generic += COMMA;
    // DATA_SIZE_OUT
    generic += to_string(nodes[nodeId].outputs.output[0].bitSize);
    generic += COMMA;
    // AMOUNT_OF_BB_IDS
    generic += to_string(nodes[nodeId].orderings.size());
    generic += COMMA;
    // AMOUNT_OF_SHARED_COMPONENTS
    generic += to_string(maxSharedComponents);
    generic += COMMA;
    // BB_ID_INFO_SIZE
    generic += to_string(bbIdInfoSize);
    generic += COMMA;
    // BB_COUNT_INFO_SIZE
    generic += to_string(bbCountInfoSize);
  } else if (nodes[nodeId].type.find("SEL") != std::string::npos) {
    generic += to_string(nodes[nodeId].inputs.size);
    generic += COMMA;
    // TODO change hardcoded number of groups
    // TODO change to number of groups
    generic += to_string(2);
    generic += COMMA;
    generic += to_string(nodes[nodeId].inputs.input[0].bitSize);
    generic += COMMA;
    generic += to_string(
        nodes[nodeId].outputs.output[nodes[nodeId].outputs.size - 1].bitSize);
  }

  return generic;
}

static void writeComponents() {
  string entity = "";
  string generic = "";
  string inputPort = "";
  string inputSignal = "";
  string outputPort = "";
  string outputSignal = "";

  for (int i = 0; i < componentsInNetlist; i++) {

    netlist << endl;

    entity = nodes[i].name;
    entity += ": entity work.";
    if (nodes[i].type == "Operator") {
      entity += nodes[i].componentOperator;
    } else if (nodes[i].type == "LSQ") {
      entity = "c_" + nodes[i].name + ":" + nodes[i].name;
    } else {
      entity += getComponentEntity(nodes[i].componentOperator, i);
    }

    if (nodes[i].type != "LSQ") {
      entity += "(arch)";

      generic = " generic map (";

      // generic += get_generic ( nodes[i].node_id );
      generic += getGeneric(i);

      generic += ")";

      netlist << entity << generic << endl;
    } else {
      netlist << entity << endl;
    }

    netlist << "port map (" << endl;

    if (nodes[i].type != "LSQ") {
      netlist << "\t"
              << "clk => " << nodes[i].name << "_clk";
      netlist << COMMA << endl
              << "\t"
              << "rst => " << nodes[i].name << "_rst";
    } else {
      netlist << "\t"
              << "clock => " << nodes[i].name << "_clk";
      netlist << COMMA << endl
              << "\t"
              << "reset => " << nodes[i].name << "_rst";

      // Andrea 20200117 Added to be compatible with chisel LSQ
      netlist << "," << endl;
      //            netlist << "\t" << "io_memIsReadyForLoads => '1' ," << endl;
      //            netlist << "\t" << "io_memIsReadyForStores => '1' ";
      netlist << "\t"
              << "io_memIsReadyForLoads => " << nodes[i].name << "_load_ready"
              << COMMA << endl;
      netlist << "\t"
              << "io_memIsReadyForStores => " << nodes[i].name
              << "_store_ready";
    }
    int indx = 0;

    //         if ( nodes[i].type == "Entry" )
    //         {
    //                 netlist << "\t" << "ap_start" << " => " << "ap_start" <<
    //                 COMMA << endl;
    //
    //                 //Write the Ready ports
    //                 input_port = "elastic_start";
    //
    //                 input_signal = nodes[i].name;
    //                 input_signal += UNDERSCORE;
    //                 input_signal += VALID_ARRAY;
    //                 input_signal += UNDERSCORE;
    //                 input_signal += "0";
    //
    //
    //                 netlist << "\t" << VALID_ARRAY << " => " << input_signal
    //                 << COMMA << endl;
    //
    //                 //Write the Ready ports
    //                 input_port = "elastic_start";
    //
    //                 input_signal = nodes[i].name;
    //                 input_signal += UNDERSCORE;
    //                 input_signal += NREADY_ARRAY;
    //                 input_signal += UNDERSCORE;
    //                 input_signal += "0";
    //                 netlist << "\t" << NREADY_ARRAY << " => " << input_signal
    //                 << endl;
    //         }
    //         else
    if (nodes[i].type == "LSQ" || nodes[i].type == "MC") {
      for (int lsqIndx = 0; lsqIndx < nodes[i].inputs.size; lsqIndx++) {
        // cout << nodes[i].name << "LSQ input "<< lsqIndx << " = " <<
        // nodes[i].inputs.input[lsqIndx].type << " port = " <<
        // nodes[i].inputs.input[lsqIndx].port << " infoType = "
        // <<nodes[i].inputs.input[lsqIndx].infoType << endl;
      }

      for (int lsqIndx = 0; lsqIndx < nodes[i].outputs.size; lsqIndx++) {
        // cout << nodes[i].name << "LSQ output "<< lsqIndx << " = " <<
        // nodes[i].outputs.output[lsqIndx].type << " port = " <<
        // nodes[i].outputs.output[lsqIndx].port << " infoType = "
        // <<nodes[i].outputs.output[lsqIndx].infoType << endl;
      }

      netlist << "," << endl;

      if (nodes[i].type == "LSQ") {
        inputSignal = nodes[i].name;
      } else {
        inputSignal = nodes[i].memory;
      }
      inputSignal += UNDERSCORE;
      inputSignal += "dout0";
      inputSignal += COMMA;

      netlist << "\t"
              << "io_storeDataOut"
              << " => " << inputSignal << endl;

      if (nodes[i].type == "LSQ") {
        inputSignal = nodes[i].name;
      } else {
        inputSignal = nodes[i].memory;
      }
      inputSignal += UNDERSCORE;
      inputSignal += "address0";
      inputSignal += COMMA;

      netlist << "\t"
              << "io_storeAddrOut"
              << " => " << inputSignal << endl;

      inputSignal = nodes[i].name;
      inputSignal += UNDERSCORE;
      inputSignal += "we0_ce0";
      inputSignal += COMMA;

      netlist << "\t"
              << "io_storeEnable"
              << " => " << inputSignal << endl;

      if (nodes[i].type == "LSQ") {
        inputSignal = nodes[i].name;
      } else {
        inputSignal = nodes[i].memory;
      }
      inputSignal += UNDERSCORE;
      inputSignal += "din1";
      inputSignal += COMMA;

      netlist << "\t"
              << "io_loadDataIn"
              << " => " << inputSignal << endl;

      if (nodes[i].type == "LSQ") {
        inputSignal = nodes[i].name;
      } else {
        inputSignal = nodes[i].memory;
      }
      inputSignal += UNDERSCORE;
      inputSignal += "address1";
      inputSignal += COMMA;

      netlist << "\t"
              << "io_loadAddrOut"
              << " => " << inputSignal << endl;

      if (nodes[i].type == "LSQ") {
        inputSignal = nodes[i].name;
      } else {
        inputSignal = nodes[i].memory;
      }
      inputSignal += UNDERSCORE;
      inputSignal += "ce1";
      // input_signal += COMMA;

      netlist << "\t"
              << "io_loadEnable"
              << " => " << inputSignal;

      string bbReadyPrev = "";
      string bbValidPrev = "";
      string bbCountPrev = "";
      string rdReadyPrev = "";
      string rdValidPrev = "";
      string rdBitsPrev = "";
      string stAdReadyPrev = "";
      string stAdValidPrev = "";
      string stAdBitsPrev = "";
      string stDataReadyPrev = "";
      string stDataValidPrev = "";
      string stDataBitsPrev = "";

      netlist << COMMA << endl;
      for (int lsqIndx = 0; lsqIndx < nodes[i].inputs.size; lsqIndx++) {
        // cout << nodes[i].name;
        // cout << " LSQ input "<< lsqIndx << " = " <<
        // nodes[i].inputs.input[lsqIndx].type << "port = " <<
        // nodes[i].inputs.input[lsqIndx].port << "infoType = "
        // <<nodes[i].inputs.input[lsqIndx].infoType << endl;

        // if ( nodes[i].inputs.input[lsqIndx].type == "c" ||
        // (nodes[i].bbcount-- > 0 ) )
        if (nodes[i].inputs.input[lsqIndx].type == "c") {
          // netlist << COMMA << endl;
          inputPort = "io";
          inputPort += UNDERSCORE;
          inputPort += "bbpValids";
          // input_port += UNDERSCORE;
          if (nodes[i].type == "MC") {
            inputPort += "(";
            inputPort += to_string(nodes[i].inputs.input[lsqIndx].port);
            inputPort += ")";

          } else {
            inputPort += UNDERSCORE;
            inputPort += to_string(nodes[i].inputs.input[lsqIndx].port);
          }

          if (nodes[i].inputs.input[lsqIndx].infoType ==
              "fake") // Andrea 20200128 Try to force 0 to inputs.
          {
            inputSignal = "'0',";
          } else {
            inputSignal = nodes[i].name;
            inputSignal += UNDERSCORE;
            inputSignal += PVALID_ARRAY;
            inputSignal += UNDERSCORE;
            inputSignal += to_string(lsqIndx);
            inputSignal += COMMA;
          }
          // netlist << "\t" << input_port << " => "  << input_signal << endl;
          bbValidPrev += "\t" + inputPort + " => " + inputSignal + "\n";

          inputPort = "io";
          inputPort += UNDERSCORE;
          inputPort += "bbReadyToPrevs";
          // input_port += UNDERSCORE;
          if (nodes[i].type == "MC") {
            inputPort += "(";
            inputPort += to_string(nodes[i].inputs.input[lsqIndx].port);
            inputPort += ")";
          } else {
            inputPort += UNDERSCORE;
            inputPort += to_string(nodes[i].inputs.input[lsqIndx].port);
          }

          inputSignal = nodes[i].name;
          inputSignal += UNDERSCORE;
          inputSignal += READY_ARRAY;
          inputSignal += UNDERSCORE;
          inputSignal += to_string(lsqIndx);
          inputSignal += COMMA;

          // netlist << "\t" << input_port << " => "  << input_signal << endl;
          bbReadyPrev += "\t" + inputPort + " => " + inputSignal + "\n";

          if (nodes[i].type == "MC") {
            inputPort = "io";
            inputPort += UNDERSCORE;
            inputPort += "bb_stCountArray";
            // input_port += UNDERSCORE;
            if (nodes[i].type == "MC") {
              inputPort += "(";
              inputPort += to_string(nodes[i].inputs.input[lsqIndx].port);
              inputPort += ")";
            } else {
              inputPort += UNDERSCORE;
              inputPort += to_string(nodes[i].inputs.input[lsqIndx].port);
            }

            if (nodes[i].inputs.input[lsqIndx].infoType ==
                "fake") // Andrea 20200128 Try to force 0 to inputs.
            {
              inputSignal = "x\"00000000\",";
            } else {

              inputSignal = nodes[i].name;
              inputSignal += UNDERSCORE;
              inputSignal += DATAIN_ARRAY;
              inputSignal += UNDERSCORE;
              inputSignal += to_string(lsqIndx);
              inputSignal += COMMA;
            }

            // netlist << "\t" << input_port << " => "  << input_signal;
            bbCountPrev += "\t" + inputPort + " => " + inputSignal + "\n";
          }

        } else if (nodes[i].inputs.input[lsqIndx].type == "l") {
          // netlist << COMMA << endl;
          // static int load_indx = 0;
          // io_rdPortsPrev_0_ready"

          if (nodes[i].type == "LSQ") {
            inputPort = "io";
            inputPort += UNDERSCORE;
            inputPort += "rdPortsPrev";
            inputPort += UNDERSCORE;
            // input_port += to_string(load_indx);
            inputPort += to_string(nodes[i].inputs.input[lsqIndx].port);

            inputPort += UNDERSCORE;
            inputPort += "ready";
          } else {
            inputPort = "io";
            inputPort += UNDERSCORE;
            inputPort += "rdPortsPrev";
            inputPort += UNDERSCORE;
            inputPort += "ready";
            inputPort += "(";
            //                    input_port += to_string(load_indx);
            inputPort += to_string(nodes[i].inputs.input[lsqIndx].port);

            inputPort += ")";
          }
          inputSignal = nodes[i].name;
          inputSignal += UNDERSCORE;
          inputSignal += READY_ARRAY;
          inputSignal += UNDERSCORE;
          inputSignal += to_string(lsqIndx);
          inputSignal += COMMA;

          rdReadyPrev += "\t" + inputPort + " => " + inputSignal + "\n";
          // netlist << "\t" << input_port << " => "  << input_signal << endl;

          if (nodes[i].type == "LSQ") {
            inputPort = "io";
            inputPort += UNDERSCORE;
            inputPort += "rdPortsPrev";
            inputPort += UNDERSCORE;
            // input_port += to_string(load_indx);
            inputPort += to_string(nodes[i].inputs.input[lsqIndx].port);

            inputPort += UNDERSCORE;
            inputPort += "valid";
          } else {
            inputPort = "io";
            inputPort += UNDERSCORE;
            inputPort += "rdPortsPrev";
            inputPort += UNDERSCORE;
            inputPort += "valid";
            inputPort += "(";
            // input_port += to_string(load_indx);
            inputPort += to_string(nodes[i].inputs.input[lsqIndx].port);

            inputPort += ")";
          }
          inputSignal = nodes[i].name;
          inputSignal += UNDERSCORE;
          inputSignal += PVALID_ARRAY;
          inputSignal += UNDERSCORE;
          inputSignal += to_string(lsqIndx);
          inputSignal += COMMA;

          rdValidPrev += "\t" + inputPort + " => " + inputSignal + "\n";
          // netlist << "\t" << input_port << " => "  << input_signal << endl;

          if (nodes[i].type == "LSQ") {
            inputPort = "io";
            inputPort += UNDERSCORE;
            inputPort += "rdPortsPrev";
            inputPort += UNDERSCORE;
            // input_port += to_string(load_indx);
            inputPort += to_string(nodes[i].inputs.input[lsqIndx].port);

            inputPort += UNDERSCORE;
            inputPort += "bits";
          } else {
            inputPort = "io";
            inputPort += UNDERSCORE;
            inputPort += "rdPortsPrev";
            inputPort += UNDERSCORE;
            inputPort += "bits";
            inputPort += "(";
            // input_port += to_string(load_indx);
            inputPort += to_string(nodes[i].inputs.input[lsqIndx].port);

            inputPort += ")";
          }
          inputSignal = nodes[i].name;
          inputSignal += UNDERSCORE;
          inputSignal += DATAIN_ARRAY;
          inputSignal += UNDERSCORE;
          inputSignal += to_string(lsqIndx);
          // input_signal += COMMA;

          rdBitsPrev += "\t" + inputPort + " => " + inputSignal + COMMA + "\n";
        } else if (nodes[i].inputs.input[lsqIndx].type == "s") {

          // netlist << COMMA << endl;
          // static int store_add_indx = 0;
          // static int store_data_indx = 0;

          if (nodes[i].type == "LSQ") {
            //"io_wrAddrPorts_0_ready"
            inputPort = "io";
            inputPort += UNDERSCORE;
            inputPort += "wr";
            if (nodes[i].inputs.input[lsqIndx].infoType == "a") {
              inputPort += "Addr";
              inputPort += "Ports";
              inputPort += UNDERSCORE;
              // input_port += to_string(store_add_indx);
              inputPort += to_string(nodes[i].inputs.input[lsqIndx].port);

            } else {
              inputPort += "Data";

              inputPort += "Ports";
              inputPort += UNDERSCORE;
              // input_port += to_string(store_data_indx);
              inputPort += to_string(nodes[i].inputs.input[lsqIndx].port);
            }

            inputPort += UNDERSCORE;
            inputPort += "valid";
          } else {
            //"io_wrAddrPorts_0_ready"
            inputPort = "io";
            inputPort += UNDERSCORE;
            inputPort += "wr";
            if (nodes[i].inputs.input[lsqIndx].infoType == "a") {
              inputPort += "Addr";
            } else {
              inputPort += "Data";
            }

            inputPort += "Ports";
            inputPort += UNDERSCORE;
            inputPort += "valid";
            inputPort += "(";
            // input_port += to_string(store_data_indx);
            inputPort += to_string(nodes[i].inputs.input[lsqIndx].port);

            inputPort += ")";
          }

          inputSignal = nodes[i].name;
          inputSignal += UNDERSCORE;
          inputSignal += PVALID_ARRAY;
          inputSignal += UNDERSCORE;
          inputSignal += to_string(lsqIndx);
          inputSignal += COMMA;

          if (nodes[i].inputs.input[lsqIndx].infoType == "a")
            stAdValidPrev += "\t" + inputPort + " => " + inputSignal + "\n";
          else
            stDataValidPrev += "\t" + inputPort + " => " + inputSignal + "\n";

          // netlist << "\t" << input_port << " => "  << input_signal << endl;

          if (nodes[i].type == "LSQ") {

            inputPort = "io";
            inputPort += UNDERSCORE;
            inputPort += "wr";
            if (nodes[i].inputs.input[lsqIndx].infoType == "a") {
              inputPort += "Addr";
              inputPort += "Ports";
              inputPort += UNDERSCORE;
              // if ( nodes[i].type == "MC" )  { input_port +="("; input_port +=
              // to_string(nodes[i].inputs.input[lsqIndx].port); input_port
              // +=")"; } else { input_port += UNDERSCORE; input_port +=
              // to_string(nodes[i].inputs.input[lsqIndx].port); } input_port
              // += to_string(store_add_indx);
              inputPort += to_string(nodes[i].inputs.input[lsqIndx].port);

            } else {
              inputPort += "Data";

              inputPort += "Ports";
              inputPort += UNDERSCORE;
              // if ( nodes[i].type == "MC" )  { input_port +="("; input_port +=
              // to_string(nodes[i].inputs.input[lsqIndx].port); input_port
              // +=")"; } else { input_port += UNDERSCORE; input_port +=
              // to_string(nodes[i].inputs.input[lsqIndx].port); } input_port
              // += to_string(store_data_indx);
              inputPort += to_string(nodes[i].inputs.input[lsqIndx].port);
            }

            inputPort += UNDERSCORE;
            inputPort += "ready";
          } else {
            inputPort = "io";
            inputPort += UNDERSCORE;
            inputPort += "wr";
            if (nodes[i].inputs.input[lsqIndx].infoType == "a") {
              inputPort += "Addr";
              inputPort += "Ports";
              inputPort += UNDERSCORE;
              inputPort += "ready";
              inputPort += "(";
              // input_port += to_string(store_add_indx);
              inputPort += to_string(nodes[i].inputs.input[lsqIndx].port);

              inputPort += ")";
            } else {
              inputPort += "Data";
              inputPort += "Ports";
              inputPort += UNDERSCORE;
              inputPort += "ready";
              inputPort += "(";
              // input_port += to_string(store_data_indx);
              inputPort += to_string(nodes[i].inputs.input[lsqIndx].port);

              inputPort += ")";
            }
          }

          inputSignal = nodes[i].name;
          inputSignal += UNDERSCORE;
          inputSignal += READY_ARRAY;
          inputSignal += UNDERSCORE;
          inputSignal += to_string(lsqIndx);
          inputSignal += COMMA;

          if (nodes[i].inputs.input[lsqIndx].infoType == "a")
            stAdReadyPrev += "\t" + inputPort + " => " + inputSignal + "\n";
          else
            stDataReadyPrev += "\t" + inputPort + " => " + inputSignal + "\n";

          // netlist << "\t" << input_port << " => "  << input_signal << endl;

          if (nodes[i].type == "LSQ") {

            inputPort = "io";
            inputPort += UNDERSCORE;
            inputPort += "wr";
            if (nodes[i].inputs.input[lsqIndx].infoType == "a") {
              inputPort += "Addr";
              inputPort += "Ports";
              inputPort += UNDERSCORE;
              // if ( nodes[i].type == "MC" )  { input_port +="("; input_port +=
              // to_string(nodes[i].inputs.input[lsqIndx].port); input_port
              // +=")"; } else { input_port += UNDERSCORE; input_port +=
              // to_string(nodes[i].inputs.input[lsqIndx].port); } input_port
              // += to_string(store_add_indx);
              inputPort += to_string(nodes[i].inputs.input[lsqIndx].port);
            } else {
              inputPort += "Data";
              inputPort += "Ports";
              inputPort += UNDERSCORE;
              // if ( nodes[i].type == "MC" )  { input_port +="("; input_port +=
              // to_string(nodes[i].inputs.input[lsqIndx].port); input_port
              // +=")"; } else { input_port += UNDERSCORE; input_port +=
              // to_string(nodes[i].inputs.input[lsqIndx].port); } input_port
              // += to_string(store_data_indx);
              inputPort += to_string(nodes[i].inputs.input[lsqIndx].port);
            }

            inputPort += UNDERSCORE;
            inputPort += "bits";

          } else {
            inputPort = "io";
            inputPort += UNDERSCORE;
            inputPort += "wr";
            if (nodes[i].inputs.input[lsqIndx].infoType == "a") {
              inputPort += "Addr";
              inputPort += "Ports";
              inputPort += UNDERSCORE;
              inputPort += "bits";
              inputPort += "(";
              // input_port += to_string(store_add_indx);
              inputPort += to_string(nodes[i].inputs.input[lsqIndx].port);

              inputPort += ")";
            } else {
              inputPort += "Data";
              inputPort += "Ports";
              inputPort += UNDERSCORE;
              inputPort += "bits";
              inputPort += "(";
              // input_port += to_string(store_data_indx);
              inputPort += to_string(nodes[i].inputs.input[lsqIndx].port);

              inputPort += ")";
            }
          }

          inputSignal = nodes[i].name;
          inputSignal += UNDERSCORE;
          inputSignal += DATAIN_ARRAY;
          inputSignal += UNDERSCORE;
          inputSignal += to_string(lsqIndx);
          // input_signal += COMMA;

          if (nodes[i].inputs.input[lsqIndx].infoType == "a")
            stAdBitsPrev +=
                "\t" + inputPort + " => " + inputSignal + COMMA + "\n";
          else
            stDataBitsPrev +=
                "\t" + inputPort + " => " + inputSignal + COMMA + "\n";

          // netlist << "\t" << input_port << " => "  << input_signal;
        }
      }

      netlist << bbReadyPrev;
      netlist << bbValidPrev;
      netlist << bbCountPrev;
      netlist << rdReadyPrev;
      netlist << rdValidPrev;
      netlist << rdBitsPrev;
      netlist << stAdReadyPrev;
      netlist << stAdValidPrev;
      netlist << stAdBitsPrev;
      netlist << stDataReadyPrev;
      netlist << stDataValidPrev;
      netlist << stDataBitsPrev;

      string rdReadyNext = "";
      string rdValidNext = "";
      string rdBitsNext = "";
      string emptyReady = "";
      string emptyValid = "";

      for (int lsqIndx = 0; lsqIndx < nodes[i].outputs.size; lsqIndx++) {
        // cout << "LSQ output "<< lsqIndx << " = " <<
        // nodes[i].outputs.output[lsqIndx].type << "port = " <<
        // nodes[i].outputs.output[lsqIndx].port << "infoType = "
        // <<nodes[i].outputs.output[lsqIndx].infoType << endl;

        if (nodes[i].outputs.output[lsqIndx].type == "c") {
          // LANA REMOVE???
          netlist << COMMA << endl;
          inputPort = "io";
          inputPort += UNDERSCORE;
          inputPort += "bbValids";
          // input_port += UNDERSCORE;
          if (nodes[i].type == "MC") {
            inputPort += "(";
            inputPort += to_string(nodes[i].inputs.input[lsqIndx].port);
            inputPort += ")";
          } else {
            inputPort += UNDERSCORE;
            inputPort += to_string(nodes[i].inputs.input[lsqIndx].port);
          }

          inputSignal = nodes[i].name;
          inputSignal += UNDERSCORE;
          inputSignal += VALID_ARRAY;
          inputSignal += UNDERSCORE;
          inputSignal += to_string(lsqIndx);
          inputSignal += COMMA;

          netlist << "\t" << inputPort << " => " << inputSignal << endl;

          inputPort = "io";
          inputPort += UNDERSCORE;
          inputPort += "bbReadyToNexts";
          inputPort += UNDERSCORE;
          if (nodes[i].type == "MC") {
            inputPort += "(";
            inputPort += to_string(nodes[i].inputs.input[lsqIndx].port);
            inputPort += ")";
          } else {
            inputPort += UNDERSCORE;
            inputPort += to_string(nodes[i].inputs.input[lsqIndx].port);
          }

          inputSignal = nodes[i].name;
          inputSignal += UNDERSCORE;
          inputSignal += NREADY_ARRAY;
          inputSignal += UNDERSCORE;
          inputSignal += to_string(lsqIndx);
          // input_signal += COMMA;

          netlist << "\t" << inputPort << " => " << inputSignal;

        } else if (nodes[i].outputs.output[lsqIndx].type == "l") {
          // static int load_indx = 0;

          // netlist << COMMA << endl;

          if (nodes[i].type == "LSQ") {

            // io_rdPortsPrev_0_ready"
            inputPort = "io";
            inputPort += UNDERSCORE;
            inputPort += "rdPortsNext";
            inputPort += UNDERSCORE;
            // if ( nodes[i].type == "MC" )  { input_port +="("; input_port +=
            // to_string(nodes[i].inputs.input[lsqIndx].port); input_port
            // +=")"; } else { input_port += UNDERSCORE; input_port +=
            // to_string(nodes[i].inputs.input[lsqIndx].port); } input_port +=
            // to_string(load_indx);
            inputPort += to_string(nodes[i].outputs.output[lsqIndx].port);

            inputPort += UNDERSCORE;
            inputPort += "ready";
          } else {
            // io_rdPortsPrev_0_ready"
            inputPort = "io";
            inputPort += UNDERSCORE;
            inputPort += "rdPortsNext";
            inputPort += UNDERSCORE;
            inputPort += "ready";
            inputPort += "(";
            inputPort += to_string(nodes[i].outputs.output[lsqIndx].port);
            inputPort += ")";
          }
          inputSignal = nodes[i].name;
          inputSignal += UNDERSCORE;
          inputSignal += NREADY_ARRAY;
          inputSignal += UNDERSCORE;
          inputSignal += to_string(lsqIndx);
          inputSignal += COMMA;

          // netlist << "\t" << input_port << " => "  << input_signal << endl;
          rdReadyNext += "\t" + inputPort + " => " + inputSignal + "\n";

          if (nodes[i].type == "LSQ") {

            inputPort = "io";
            inputPort += UNDERSCORE;
            inputPort += "rdPortsNext";
            inputPort += UNDERSCORE;
            // if ( nodes[i].type == "MC" )  { input_port +="("; input_port +=
            // to_string(nodes[i].inputs.input[lsqIndx].port); input_port
            // +=")"; } else { input_port += UNDERSCORE; input_port +=
            // to_string(nodes[i].inputs.input[lsqIndx].port); }
            inputPort += to_string(nodes[i].outputs.output[lsqIndx].port);

            inputPort += UNDERSCORE;
            inputPort += "valid";
          } else {
            inputPort = "io";
            inputPort += UNDERSCORE;
            inputPort += "rdPortsNext";
            inputPort += UNDERSCORE;
            inputPort += "valid";
            inputPort += "(";
            inputPort += to_string(nodes[i].outputs.output[lsqIndx].port);
            inputPort += ")";
          }
          inputSignal = nodes[i].name;
          inputSignal += UNDERSCORE;
          inputSignal += VALID_ARRAY;
          inputSignal += UNDERSCORE;
          inputSignal += to_string(lsqIndx);
          inputSignal += COMMA;

          // netlist << "\t" << input_port << " => "  << input_signal << endl;
          rdValidNext += "\t" + inputPort + " => " + inputSignal + "\n";

          if (nodes[i].type == "LSQ") {

            inputPort = "io";
            inputPort += UNDERSCORE;
            inputPort += "rdPortsNext";
            inputPort += UNDERSCORE;
            // if ( nodes[i].type == "MC" )  { input_port +="("; input_port +=
            // to_string(nodes[i].inputs.input[lsqIndx].port); input_port
            // +=")"; } else { input_port += UNDERSCORE; input_port +=
            // to_string(nodes[i].inputs.input[lsqIndx].port); }
            inputPort += to_string(nodes[i].outputs.output[lsqIndx].port);

            inputPort += UNDERSCORE;
            inputPort += "bits";
          } else {
            inputPort = "io";
            inputPort += UNDERSCORE;
            inputPort += "rdPortsNext";
            inputPort += UNDERSCORE;
            inputPort += "bits";
            inputPort += "(";
            inputPort += to_string(nodes[i].outputs.output[lsqIndx].port);
            inputPort += ")";
          }
          inputSignal = nodes[i].name;
          inputSignal += UNDERSCORE;
          inputSignal += DATAOUT_ARRAY;
          inputSignal += UNDERSCORE;
          inputSignal += to_string(lsqIndx);
          inputSignal += COMMA;

          // netlist << "\t" << input_port << " => "  << input_signal;
          rdBitsNext += "\t" + inputPort + " => " + inputSignal + "\n";

        } else if (nodes[i].outputs.output[lsqIndx].type == "s") {
          // LANA REMOVE???
          netlist << COMMA << endl;
          inputPort = "io";
          inputPort += UNDERSCORE;
          inputPort += "wrpValids";
          inputPort += UNDERSCORE;
          // if ( nodes[i].type == "MC" )  { input_port +="("; input_port +=
          // to_string(nodes[i].inputs.input[lsqIndx].port); input_port +=")";
          // } else { input_port += UNDERSCORE; input_port +=
          // to_string(nodes[i].inputs.input[lsqIndx].port); } input_port +=
          // to_string(store_indx);
          inputPort += to_string(nodes[i].outputs.output[lsqIndx].port);

          inputSignal = nodes[i].name;
          inputSignal += UNDERSCORE;
          inputSignal += VALID_ARRAY;
          inputSignal += UNDERSCORE;
          inputSignal += to_string(lsqIndx);
          inputSignal += COMMA;

          netlist << "\t" << inputPort << " => " << inputSignal << endl;

          inputPort = "io";
          inputPort += UNDERSCORE;
          inputPort += "wrReadyToPrevs";
          inputPort += UNDERSCORE;
          // if ( nodes[i].type == "MC" )  { input_port +="("; input_port +=
          // to_string(nodes[i].inputs.input[lsqIndx].port); input_port +=")";
          // } else { input_port += UNDERSCORE; input_port +=
          // to_string(nodes[i].inputs.input[lsqIndx].port); } input_port +=
          // to_string(store_indx);
          inputPort += to_string(nodes[i].outputs.output[lsqIndx].port);

          inputSignal = nodes[i].name;
          inputSignal += UNDERSCORE;
          inputSignal += NREADY_ARRAY;
          inputSignal += UNDERSCORE;
          inputSignal += to_string(lsqIndx);
          // input_signal += COMMA;

          netlist << "\t" << inputPort << " => " << inputSignal;
        } else if (nodes[i].outputs.output[lsqIndx].type == "e") {
          inputPort = "io";
          inputPort += UNDERSCORE;
          inputPort += "Empty_Valid";

          inputSignal = nodes[i].name;
          inputSignal += UNDERSCORE;
          inputSignal += VALID_ARRAY;
          inputSignal += UNDERSCORE;
          inputSignal += to_string(lsqIndx);

          if (nodes[i].type !=
              "LSQ") // Andrea 20200117 Added to be compatible with chisel LSQ
            inputSignal += COMMA;

          // netlist << "\t" << input_port << " => "  << input_signal << endl;
          emptyValid += "\t" + inputPort + " => " + inputSignal + "\n";

          inputPort = "io";
          inputPort += UNDERSCORE;
          inputPort += "Empty_Ready";

          inputSignal = nodes[i].name;
          inputSignal += UNDERSCORE;
          inputSignal += NREADY_ARRAY;
          inputSignal += UNDERSCORE;
          inputSignal += to_string(lsqIndx);
          // input_signal += COMMA;

          // netlist << "\t" << input_port << " => "  << input_signal;
          emptyReady += "\t" + inputPort + " => " + inputSignal + "\n";
        }
      }

      netlist << rdReadyNext;
      netlist << rdValidNext;
      netlist << rdBitsNext;
      netlist << emptyValid;

      if (nodes[i].type != "LSQ")
        netlist << emptyReady;

    } else if (nodes[i].type == "Exit") {

      for (indx = 0; indx < nodes[i].inputs.size; indx++) {

        if (nodes[i].inputs.input[indx].type != "e") {
          inputPort = componentsType[0].inPortsNameStr[0];
          inputPort += "(";
          inputPort += to_string(indx - getMemoryInputs(i));
          inputPort += ")";

          inputSignal = nodes[i].name;
          inputSignal += UNDERSCORE;
          inputSignal +=
              componentsType[nodes[i].componentType].inPortsNameStr[0];
          inputSignal += UNDERSCORE;
          inputSignal += to_string(indx);
          netlist << COMMA << endl
                  << "\t" << inputPort << " => " << inputSignal;
        }
      }
      for (indx = 0; indx < nodes[i].inputs.size; indx++) {

        if (nodes[i].inputs.input[indx].type != "e") {
          // Write the Ready ports
          inputPort = PVALID_ARRAY;
          inputPort += "(";
          inputPort += to_string(indx - getMemoryInputs(i));
          inputPort += ")";
        } else {
          // Write the Ready ports
          inputPort = "eValidArray";
          inputPort += "(";
          inputPort += to_string(indx);
          inputPort += ")";
        }

        // if ( indx == ( nodes[i].inputs.size - 1 ) )
        {
          inputSignal = nodes[i].name;
          inputSignal += UNDERSCORE;
          inputSignal += PVALID_ARRAY;
          inputSignal += UNDERSCORE;
          inputSignal += to_string(indx);
        }
        // else
        {
          //    input_signal = "\'0\', --Andrea forced to 0 to run the
          //    simulation";
        }
        netlist << COMMA << endl << "\t" << inputPort << " => " << inputSignal;
      }
      for (indx = 0; indx < nodes[i].inputs.size; indx++) {
        if (nodes[i].inputs.input[indx].type != "e") {
          // Write the Ready ports
          inputPort = READY_ARRAY;
          inputPort += "(";
          inputPort += to_string(indx - getMemoryInputs(i));
          inputPort += ")";
        } else {
          // Write the Ready ports
          inputPort = "eReadyArray";
          inputPort += "(";
          inputPort += to_string(indx);
          inputPort += ")";
        }
        inputSignal = nodes[i].name;
        inputSignal += UNDERSCORE;
        inputSignal += READY_ARRAY;
        inputSignal += UNDERSCORE;
        inputSignal += to_string(indx);

        netlist << COMMA << endl << "\t" << inputPort << " => " << inputSignal;
      }

      // netlist << COMMA << endl << "\t" << "ap_done" << " => " << "ap_done";

      inputPort = componentsType[0].outPortsNameStr[0];
      inputPort += "(";
      inputPort += "0";
      inputPort += ")";

      inputSignal = nodes[i].name;
      inputSignal += UNDERSCORE;
      inputSignal += componentsType[nodes[i].componentType].outPortsNameStr[0];
      inputSignal += UNDERSCORE;
      inputSignal += "0";

      netlist << COMMA << endl << "\t" << inputPort << " => " << inputSignal;

      inputPort = VALID_ARRAY;
      inputPort += "(";
      inputPort += "0";
      inputPort += ")";

      inputSignal = nodes[i].name;
      inputSignal += UNDERSCORE;
      inputSignal += VALID_ARRAY;
      inputSignal += UNDERSCORE;
      inputSignal += "0";

      netlist << COMMA << endl << "\t" << inputPort << " => " << inputSignal;

      inputPort = NREADY_ARRAY;
      inputPort += "(";
      inputPort += "0";
      inputPort += ")";

      inputSignal = nodes[i].name;
      inputSignal += UNDERSCORE;
      inputSignal += NREADY_ARRAY;
      inputSignal += UNDERSCORE;
      inputSignal += "0";

      netlist << COMMA << endl << "\t" << inputPort << " => " << inputSignal;

    } else {
      for (indx = 0; indx < nodes[i].inputs.size; indx++) {
        for (int inPortIndx = 0; inPortIndx < 1; inPortIndx++) {
          if ((nodes[i].type.find("Branch") != std::string::npos &&
               indx == 1) ||
              ((nodes[i].componentOperator.find("select") !=
                std::string::npos) &&
               indx == 0) ||
              ((nodes[i].componentOperator.find("Mux") != std::string::npos) &&
               indx == 0) ||
              (nodes[i].type.find(COMPONENT_DISTRIBUTOR) != std::string::npos &&
               indx == 1)) {
            inputPort = "Condition(0)";
          } else if (nodes[i].type.find("Selector") != std::string::npos &&
                     indx >= nodes[i].inputs.size -
                                 (int)nodes[i].orderings.size()) {
            inputPort = "bbInfoData(";
            inputPort += to_string(
                indx - (nodes[i].inputs.size - nodes[i].orderings.size()));
            inputPort += ")";
          }
          // Lana 9.6.2021. Changed lsq memory port interface
          else if (((nodes[i].componentOperator.find("mc_store_op") !=
                     std::string::npos) ||
                    (nodes[i].componentOperator.find("mc_load_op") !=
                     std::string::npos) ||
                    (nodes[i].componentOperator.find("lsq_store_op") !=
                     std::string::npos) ||
                    (nodes[i].componentOperator.find("lsq_load_op") !=
                     std::string::npos)) &&
                   indx == 1) {
            inputPort = "input_addr";
          } else {
            inputPort = componentsType[0].inPortsNameStr[inPortIndx];
            inputPort += "(";
            if ((nodes[i].componentOperator.find("select") !=
                 std::string::npos) ||
                ((nodes[i].componentOperator.find("Mux") !=
                  std::string::npos))) {
              inputPort += to_string(indx - 1);
            } else {
              inputPort += to_string(indx);
            }
            inputPort += ")";
          }

          /*
          if ( nodes[i].inputs.input[indx].type == "i" )
          {
              input_signal = "not "; //inverted
              input_signal += nodes[i].name;
          }
          else
              input_signal = nodes[i].name;
          */
          inputSignal = nodes[i].name;
          inputSignal += UNDERSCORE;
          inputSignal +=
              componentsType[nodes[i].componentType].inPortsNameStr[inPortIndx];
          inputSignal += UNDERSCORE;
          inputSignal += to_string(indx);

          netlist << COMMA << endl
                  << "\t" << inputPort << " => " << inputSignal;
        }
      }

      for (indx = 0; indx < nodes[i].inputs.size; indx++) {
        if (nodes[i].type.find("Selector") != std::string::npos &&
            indx >= nodes[i].inputs.size - (int)nodes[i].orderings.size()) {
          // ctrlForks ports have another name
          inputPort = "bbInfoPValid";
          inputPort += "(";
          inputPort += to_string(
              indx - (nodes[i].inputs.size - nodes[i].orderings.size()));
          inputPort += ")";
        }

        else {
          // Write the Ready ports
          inputPort = PVALID_ARRAY;
          inputPort += "(";
          inputPort += to_string(indx);
          inputPort += ")";
        }

        inputSignal = nodes[i].name;
        inputSignal += UNDERSCORE;
        inputSignal += PVALID_ARRAY;
        inputSignal += UNDERSCORE;
        inputSignal += to_string(indx);

        netlist << COMMA << endl << "\t" << inputPort << " => " << inputSignal;
      }

      for (indx = 0; indx < nodes[i].inputs.size; indx++) {
        if (nodes[i].type.find("Selector") != std::string::npos &&
            indx >= nodes[i].inputs.size - (int)nodes[i].orderings.size()) {
          // ctrlForks ports have another name
          inputPort = "bbInfoReady";
          inputPort += "(";
          inputPort += to_string(
              indx - (nodes[i].inputs.size - nodes[i].orderings.size()));
          inputPort += ")";
        } else {
          // Write the Ready ports
          inputPort = READY_ARRAY;
          inputPort += "(";
          inputPort += to_string(indx);
          inputPort += ")";
        }
        inputSignal = nodes[i].name;
        inputSignal += UNDERSCORE;
        inputSignal += READY_ARRAY;
        inputSignal += UNDERSCORE;
        inputSignal += to_string(indx);

        netlist << COMMA << endl << "\t" << inputPort << " => " << inputSignal;
      }

      // if ( nodes[i].name.find("load") != std::string::npos )
      if (nodes[i].componentOperator == "load_op") {
        inputPort = "read_enable";

        inputSignal = nodes[i].name;
        inputSignal += UNDERSCORE;
        inputSignal += "read_enable";

        netlist << COMMA << endl << "\t" << inputPort << " => " << inputSignal;

        inputPort = "read_address";

        inputSignal = nodes[i].name;
        inputSignal += UNDERSCORE;
        inputSignal += "read_address";

        netlist << COMMA << endl << "\t" << inputPort << " => " << inputSignal;

        inputPort = "data_from_memory";

        inputSignal = nodes[i].name;
        inputSignal += UNDERSCORE;
        inputSignal += "data_from_memory";

        netlist << COMMA << endl << "\t" << inputPort << " => " << inputSignal;
      }

      if (nodes[i].componentOperator == "store_op") {
        inputPort = "write_enable";

        inputSignal = nodes[i].name;
        inputSignal += UNDERSCORE;
        inputSignal += "write_enable";

        netlist << COMMA << endl << "\t" << inputPort << " => " << inputSignal;

        inputPort = "write_address";

        inputSignal = nodes[i].name;
        inputSignal += UNDERSCORE;
        inputSignal += "write_address";

        netlist << COMMA << endl << "\t" << inputPort << " => " << inputSignal;

        inputPort = "data_to_memory";

        inputSignal = nodes[i].name;
        inputSignal += UNDERSCORE;
        inputSignal += "data_to_memory";

        netlist << COMMA << endl << "\t" << inputPort << " => " << inputSignal;
      }

      for (indx = 0; indx < nodes[i].outputs.size; indx++) {

        // Write the Ready ports
        inputPort = NREADY_ARRAY;
        inputPort += "(";
        inputPort += to_string(indx);
        inputPort += ")";

        inputSignal = nodes[i].name;
        inputSignal += UNDERSCORE;
        inputSignal += NREADY_ARRAY;
        inputSignal += UNDERSCORE;
        inputSignal += to_string(indx);

        netlist << COMMA << endl << "\t" << inputPort << " => " << inputSignal;
      }
      for (indx = 0; indx < nodes[i].outputs.size; indx++) {

        // Write the Ready ports
        inputPort = VALID_ARRAY;
        inputPort += "(";
        inputPort += to_string(indx);
        inputPort += ")";

        inputSignal = nodes[i].name;
        inputSignal += UNDERSCORE;
        inputSignal += VALID_ARRAY;
        inputSignal += UNDERSCORE;
        inputSignal += to_string(indx);

        netlist << COMMA << endl << "\t" << inputPort << " => " << inputSignal;
      }
      for (indx = 0; indx < nodes[i].outputs.size; indx++) {

        for (int outPortIndx = 0;
             outPortIndx < componentsType[nodes[i].componentType].outPorts;
             outPortIndx++) {

          if ((nodes[i].type.find(COMPONENT_CTRLMERGE) != std::string::npos &&
               indx == 1) ||
              (nodes[i].type.find(COMPONENT_SEL) != std::string::npos &&
               indx == nodes[i].outputs.size - 1) ||
              (nodes[i].type.find(COMPONENT_SELECTOR) != std::string::npos &&
               indx == nodes[i].outputs.size - 1)) {
            outputPort = "Condition(0)";
          } else if (((nodes[i].componentOperator.find("mc_store_op") !=
                       std::string::npos) ||
                      (nodes[i].componentOperator.find("mc_load_op") !=
                       std::string::npos) ||
                      (nodes[i].componentOperator.find("lsq_store_op") !=
                       std::string::npos) ||
                      (nodes[i].componentOperator.find("lsq_load_op") !=
                       std::string::npos)) &&
                     indx == 1) {
            outputPort = "output_addr";
          } else {
            outputPort = componentsType[nodes[i].componentType]
                             .outPortsNameStr[outPortIndx];
            outputPort += "(";
            outputPort += to_string(indx);
            outputPort += ")";
          }

          outputSignal = nodes[i].name;
          outputSignal += UNDERSCORE;
          outputSignal += componentsType[nodes[i].componentType]
                              .outPortsNameStr[outPortIndx];
          outputSignal += UNDERSCORE;
          outputSignal += to_string(indx);
          netlist << COMMA << endl
                  << "\t" << outputPort << " => " << outputSignal;
        }
      }
    }

    if (nodes[i].type.find("Selector") != std::string::npos) {

      int maxSharedComponents = -1;
      int amountSharedComponents = 0;
      for (const auto &orderingPerBb : nodes[i].orderings) {
        int size = orderingPerBb.size();
        amountSharedComponents += size;
        if (maxSharedComponents < size) {
          maxSharedComponents = size;
        }
      }
      int indexSize = ceil(log2(amountSharedComponents));

      for (size_t bbIndex = 0; bbIndex < nodes[i].orderings.size(); ++bbIndex) {
        for (int compIndex = 0; compIndex < maxSharedComponents; ++compIndex) {
          inputPort = "bbOrderingData";
          inputPort += "(";
          inputPort += to_string(bbIndex);
          inputPort += ")(";
          inputPort += to_string(compIndex);
          inputPort += ")";

          int value;
          if ((size_t)compIndex < nodes[i].orderings[bbIndex].size()) {
            value = nodes[i].orderings[bbIndex][compIndex];
          } else {
            value = 0;
          }
          inputSignal = "\"";
          inputSignal += stringConstant(value, indexSize);
          inputSignal += "\"";

          netlist << COMMA << endl
                  << "\t" << inputPort << " => " << inputSignal;
        }
      }
    }
    netlist << endl << ");" << endl;
  }
}

static int getEndBitsize(void) {
  int bitsize = 0;
  for (int i = 0; i < componentsInNetlist; i++) {
    if (nodes[i].type == "Exit") {
      bitsize = (nodes[i].outputs.output[0].bitSize > 1)
                    ? (nodes[i].outputs.output[0].bitSize - 1)
                    : 0;
    }
  }
  return bitsize;
}

static void writeEntity(const string &filename) {

  string entity = cleanEntity(filename);

  string inputPort, outputPort, inputSignal, outputSignal, signal;
  netlist << "entity " << entity << " is " << endl;
  netlist << "port (" << endl;
  netlist << "\t"
          << "clk: "
          << " in std_logic;" << endl;
  netlist << "\t"
          << "rst: "
          << " in std_logic;" << endl;

  netlist << "\t"
          << "start_in: "
          << " in std_logic_vector (0 downto 0);" << endl;
  netlist << "\t"
          << "start_valid: "
          << " in std_logic;" << endl;
  netlist << "\t"
          << "start_ready: "
          << " out std_logic;" << endl;

  netlist << "\t"
          << "end_out: "
          << " out std_logic_vector (" << getEndBitsize() << " downto 0);"
          << endl;
  netlist << "\t"
          << "end_valid: "
          << " out std_logic;" << endl;
  netlist << "\t"
          << "end_ready: "
          << " in std_logic";

  for (int i = 0; i < componentsInNetlist; i++) {
    if ((nodes[i].name.find("Arg") != std::string::npos) ||
        ((nodes[i].type.find("Entry") != std::string::npos) &&
         (!(nodes[i].name.find("start") != std::string::npos)))) {
      netlist << ";" << endl;
      netlist << "\t" << nodes[i].name
              << "_din : in std_logic_vector (31 downto 0);" << endl;
      netlist << "\t" << nodes[i].name << "_valid_in : in std_logic;" << endl;
      netlist << "\t" << nodes[i].name << "_ready_out : out std_logic";
    }

    // if ( nodes[i].name.find("load") != std::string::npos )
    if (nodes[i].componentOperator == "load_op") {
      netlist << ";" << endl;
      netlist << "\t" << nodes[i].name
              << "_data_from_memory : in std_logic_vector (31 downto 0);"
              << endl;
      netlist << "\t" << nodes[i].name << "_read_enable : out std_logic;"
              << endl;
      netlist << "\t" << nodes[i].name
              << "_read_address : out std_logic_vector (31 downto 0)";
    }

    // if ( nodes[i].name.find("store") != std::string::npos )
    if (nodes[i].componentOperator == "store_op")

    {
      netlist << ";" << endl;
      netlist << "\t" << nodes[i].name
              << "_data_to_memory : out std_logic_vector (31 downto 0);"
              << endl;
      netlist << "\t" << nodes[i].name << "_write_enable : out std_logic;"
              << endl;
      netlist << "\t" << nodes[i].name
              << "_write_address : out std_logic_vector (31 downto 0)";
    }

    bool mcLsq = false;
    if (nodes[i].type.find("LSQ") != std::string::npos) {
      for (int indx = 0; indx < nodes[i].inputs.size; indx++) {
        if (nodes[i].inputs.input[indx].type == "x") {
          mcLsq = true;
          break;
        }
      }
    }
    if ((nodes[i].type.find("LSQ") != std::string::npos && !mcLsq) ||
        nodes[i].type.find("MC") != std::string::npos) {
      netlist << ";" << endl;
      netlist << "\t" << nodes[i].memory
              << "_address0 : out std_logic_vector (31 downto 0);" << endl;
      netlist << "\t" << nodes[i].memory << "_ce0 : out std_logic;" << endl;
      netlist << "\t" << nodes[i].memory << "_we0 : out std_logic;" << endl;
      netlist << "\t" << nodes[i].memory
              << "_dout0 : out std_logic_vector (31 downto 0);" << endl;
      netlist << "\t" << nodes[i].memory
              << "_din0 : in std_logic_vector (31 downto 0);" << endl;

      netlist << "\t" << nodes[i].memory
              << "_address1 : out std_logic_vector (31 downto 0);" << endl;
      netlist << "\t" << nodes[i].memory << "_ce1 : out std_logic;" << endl;
      netlist << "\t" << nodes[i].memory << "_we1 : out std_logic;" << endl;
      netlist << "\t" << nodes[i].memory
              << "_dout1 : out std_logic_vector (31 downto 0);" << endl;
      netlist << "\t" << nodes[i].memory
              << "_din1 : in std_logic_vector (31 downto 0)";
    }
  }

  netlist << ");" << endl;
  netlist << "end;" << endl << endl;
}

static void writeIntro() {

  time_t now = time(nullptr);
  char *dt = ctime(&now);

  netlist << "-- =============================================================="
          << endl;
  netlist << "-- Generated by export-vhdl" << endl;
  netlist << "-- File created: " << dt << endl;
  netlist << "-- =============================================================="
          << endl;

  netlist << "library IEEE; " << endl;
  netlist << "use IEEE.std_logic_1164.all; " << endl;
  netlist << "use IEEE.numeric_std.all; " << endl;
  netlist << "use work.customTypes.all; " << endl;

  netlist << "-- =============================================================="
          << endl;
}

static void writeLSQSignal(const std::string &name, bool isInput,
                           const std::string &type, bool isFinalSignal) {
  netlist << "\t" << name << " : ";
  if (isInput) {
    netlist << "in ";
  } else {
    netlist << "out ";
  }
  netlist << type;
  if (!isFinalSignal) {
    netlist << ";";
  }
  netlist << endl;
}

static void writeLSQDeclaration() {
  for (int i = 0; i < componentsInNetlist; i++) {
    if (nodes[i].type == "LSQ") {
      netlist << endl;
      netlist << "component " << nodes[i].name << endl;
      netlist << "port(" << endl;
      writeLSQSignal("clock", true, "std_logic", false);
      writeLSQSignal("reset", true, "std_logic", false);
      writeLSQSignal("io_memIsReadyForLoads", true, "std_logic", false);
      writeLSQSignal("io_memIsReadyForStores", true, "std_logic", false);
      writeLSQSignal("io_storeDataOut", false,
                     "std_logic_vector(" + to_string(getLSQDataWidth() - 1) +
                         " downto 0)",
                     false);
      writeLSQSignal("io_storeAddrOut", false,
                     "std_logic_vector(" + to_string(nodes[i].addressSize - 1) +
                         " downto 0)",
                     false);
      writeLSQSignal("io_storeEnable", false, "std_logic", false);

      std::string name = "";

      for (int lsqIndx = 0; lsqIndx < nodes[i].inputs.size; lsqIndx++) {
        if (nodes[i].inputs.input[lsqIndx].type == "c") {
          name =
              "io_bbpValids_" + to_string(nodes[i].inputs.input[lsqIndx].port);
          writeLSQSignal(name, true, "std_logic", false);
          name = "io_bbReadyToPrevs_" +
                 to_string(nodes[i].inputs.input[lsqIndx].port);
          writeLSQSignal(name, false, "std_logic", false);

        } else if (nodes[i].inputs.input[lsqIndx].type == "l") {
          name = "io_rdPortsPrev_" +
                 to_string(nodes[i].inputs.input[lsqIndx].port) + "_ready";
          writeLSQSignal(name, false, "std_logic", false);
          name = "io_rdPortsPrev_" +
                 to_string(nodes[i].inputs.input[lsqIndx].port) + "_valid";
          writeLSQSignal(name, true, "std_logic", false);
          name = "io_rdPortsPrev_" +
                 to_string(nodes[i].inputs.input[lsqIndx].port) + "_bits";
          // Lana 9.6.2021. rdPortsPrev is address port, set to address size
          writeLSQSignal(name, true,
                         "std_logic_vector(" +
                             to_string(nodes[i].addressSize - 1) + " downto 0)",
                         false);
        } else if (nodes[i].inputs.input[lsqIndx].type == "s") {
          name = "io_wr";
          if (nodes[i].inputs.input[lsqIndx].infoType == "a") {
            name += "AddrPorts_";
          } else {
            name += "DataPorts_";
          }
          name += to_string(nodes[i].inputs.input[lsqIndx].port) + "_valid";
          writeLSQSignal(name, true, "std_logic", false);

          name = "io_wr";
          if (nodes[i].inputs.input[lsqIndx].infoType == "a") {
            name += "AddrPorts_";
          } else {
            name += "DataPorts_";
          }
          name += to_string(nodes[i].inputs.input[lsqIndx].port) + "_ready";
          writeLSQSignal(name, false, "std_logic", false);

          name = "io_wr";
          if (nodes[i].inputs.input[lsqIndx].infoType == "a") {
            name += "AddrPorts_";
            name += to_string(nodes[i].inputs.input[lsqIndx].port) + "_bits";
            // write_lsq_signal(name, true, "std_logic_vector("+
            // to_string(get_lsq_addresswidth()-1) +" downto 0)", false);
            writeLSQSignal(name, true,
                           "std_logic_vector(" +
                               to_string(nodes[i].addressSize - 1) +
                               " downto 0)",
                           false);
          } else {
            name += "DataPorts_";
            name += to_string(nodes[i].inputs.input[lsqIndx].port) + "_bits";
            writeLSQSignal(name, true,
                           "std_logic_vector(" +
                               to_string(getLSQDataWidth() - 1) + " downto 0)",
                           false);
          }
        }
      }

      for (int lsqIndx = 0; lsqIndx < nodes[i].outputs.size; lsqIndx++) {
        if (nodes[i].outputs.output[lsqIndx].type == "c") {
          // name =
          // "io_bbValids_"+to_string(nodes[i].inputs.input[lsqIndx].port);
          name =
              "io_bbValids_" + to_string(nodes[i].outputs.output[lsqIndx].port);
          writeLSQSignal(name, false, "std_logic", false);
          // name =
          // "io_bbReadyToNexts_"+to_string(nodes[i].inputs.input[lsqIndx].port);
          name = "io_bbReadyToNexts_" +
                 to_string(nodes[i].outputs.output[lsqIndx].port);
          writeLSQSignal(name, false, "std_logic", false);
        } else if (nodes[i].outputs.output[lsqIndx].type == "l") {
          // name =
          // "io_rdPortsNext_"+to_string(nodes[i].inputs.input[lsqIndx].port) +
          // "_ready";
          name = "io_rdPortsNext_" +
                 to_string(nodes[i].outputs.output[lsqIndx].port) + "_ready";
          writeLSQSignal(name, true, "std_logic", false);
          // name =
          // "io_rdPortsNext_"+to_string(nodes[i].inputs.input[lsqIndx].port) +
          // "_valid";
          name = "io_rdPortsNext_" +
                 to_string(nodes[i].outputs.output[lsqIndx].port) + "_valid";
          writeLSQSignal(name, false, "std_logic", false);
          // name =
          // "io_rdPortsNext_"+to_string(nodes[i].inputs.input[lsqIndx].port) +
          // "_bits";
          name = "io_rdPortsNext_" +
                 to_string(nodes[i].outputs.output[lsqIndx].port) + "_bits";
          writeLSQSignal(name, false,
                         "std_logic_vector(" +
                             to_string(getLSQDataWidth() - 1) + " downto 0)",
                         false);
        } else if (nodes[i].outputs.output[lsqIndx].type == "s") {
          // name =
          // "io_wrpValids_"+to_string(nodes[i].inputs.input[lsqIndx].port);
          name = "io_wrpValids_" +
                 to_string(nodes[i].outputs.output[lsqIndx].port);
          writeLSQSignal(name, true, "std_logic", false);
          // name =
          // "io_wrReadyToPrevs_"+to_string(nodes[i].inputs.input[lsqIndx].port);
          name = "io_wrReadyToPrevs_" +
                 to_string(nodes[i].outputs.output[lsqIndx].port);
          writeLSQSignal(name, false, "std_logic", false);
        } else if (nodes[i].outputs.output[lsqIndx].type == "e") {
          writeLSQSignal("io_Empty_Valid", false, "std_logic", false);
        }
      }

      writeLSQSignal("io_loadDataIn", true,
                     "std_logic_vector(" + to_string(getLSQDataWidth() - 1) +
                         "  downto 0)",
                     false);
      // write_lsq_signal("io_loadAddrOut", false, "std_logic_vector("+
      // to_string(get_lsq_addresswidth()-1) +"  downto 0)", false);
      writeLSQSignal("io_loadAddrOut", false,
                     "std_logic_vector(" + to_string(nodes[i].addressSize - 1) +
                         "  downto 0)",
                     false);
      writeLSQSignal("io_loadEnable", false, "std_logic", true);
      netlist << ");" << endl;
      netlist << "end component;" << endl;
    }
  }
}

void writeVHDL(const string &kernelName, const string &vhdPath) {
  string entity = cleanEntity(kernelName);

  componentsType[COMPONENT_GENERIC].inPorts = 2;
  componentsType[COMPONENT_GENERIC].outPorts = 1;
  componentsType[COMPONENT_GENERIC].inPortsNameStr = inPortsNameGeneric;
  componentsType[COMPONENT_GENERIC].inPortsTypeStr = inPortsTypeGeneric;
  componentsType[COMPONENT_GENERIC].outPortsNameStr = outPortsNameGeneric;
  componentsType[COMPONENT_GENERIC].outPortsTypeStr = outPortsTypeGeneric;

  componentsType[COMPONENT_CONSTANT].inPorts = 1;
  componentsType[COMPONENT_CONSTANT].outPorts = 1;
  componentsType[COMPONENT_CONSTANT].inPortsNameStr = inPortsNameGeneric;
  componentsType[COMPONENT_CONSTANT].inPortsTypeStr = inPortsTypeGeneric;
  componentsType[COMPONENT_CONSTANT].outPortsNameStr = outPortsNameGeneric;
  componentsType[COMPONENT_CONSTANT].outPortsTypeStr = outPortsTypeGeneric;

  netlist.open(vhdPath);

  writeIntro();
  writeEntity(kernelName);
  netlist << "architecture behavioral of " << entity << " is " << endl;
  writeSignals();
  writeLSQDeclaration();
  netlist << endl << "begin" << endl;
  writeConnections();
  writeComponents();
  netlist << endl << "end behavioral; " << endl;

  netlist.close();
}
