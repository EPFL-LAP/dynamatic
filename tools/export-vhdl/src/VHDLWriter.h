//===- vhdl_writer.h - Generate VHDL corresponding to DOT -------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EXPORT_VHDL_VHDL_WRITER_H
#define EXPORT_VHDL_VHDL_WRITER_H

#include <string>

#define COMPONENT_GENERIC 0
#define COMPONENT_CONSTANT 1

#define CLK "clk"
#define RST "rst"
#define DATAIN_ARRAY "dataInArray"
#define DATAOUT_ARRAY "dataOutArray"
#define PVALID_ARRAY "pValidArray"
#define NREADY_ARRAY "nReadyArray"
#define READY_ARRAY "readyArray"
#define VALID_ARRAY "validArray"

#define MAX_COMPONENTS 6

#define ENTITY_MERGE "merge"
#define ENTITY_READ_MEMORY "read_memory_single"
#define ENTITY_WRITE_MEMORY "write_memory_single"
#define ENTITY_SINGLE_OP "SingleOp"
#define ENTITY_GET_PTR "getptr"
#define ENTITY_INT_MUL "intMul"
#define ENTITY_INT_ADD "intAdd"
#define ENTITY_INT_SUB "intSub"
#define ENTITY_BUF "elasticBuffer"
#define ENTITY_TEHB "TEHB"
#define ENTITY_OEHB "OEHB"
#define ENTITY_FIFO "elasticFifo"
#define ENTITY_NFIFO "nontranspFifo"
#define ENTITY_TFIFO "transpFifo"
#define ENTITY_FORK "fork"
#define ENTITY_LFORK "lazyfork"
#define ENTITY_ICMP "intCmp"
#define ENTITY_CONSTANT "Const"
#define ENTITY_BRANCH "Branch"
#define ENTITY_END "end_node"
#define ENTITY_START "start_node"
#define ENTITY_SOURCE "source"
#define ENTITY_SINK "sink"
#define ENTITY_MUX "Mux"
#define ENTITY_CTRLMERGE "CntrlMerge"
#define ENTITY_LSQ "LSQ"
#define ENTITY_MC "MemCont"
#define ENTITY_SEL "SEL"
#define ENTITY_DISTRIBUTOR "Distributor"
#define ENTITY_SELECTOR "Selector"
#define ENTITY_INJECTOR "Inj"
#define ENTITY_SPEC_BRANCH "speculating_branch"
#define ENTITY_COMMIT_UNIT "spec_commit"
#define ENTITY_SAVE_COMMIT_UNIT "spec_save_commit"
#define ENTITY_SAVE_UNIT "spec_save"
#define ENTITY_SPECULATOR "speculator"

#define COMPONENT_MERGE "Merge"
#define COMPONENT_READ_MEMORY "load"
#define COMPONENT_WRITE_MEMORY "store"
#define COMPONENT_SINGLE_OP "Merge"
#define COMPONENT_GET_PTR "Merge"
#define COMPONENT_INT_MUL "mul"
#define COMPONENT_INT_ADD "add"
#define COMPONENT_INT_SUB "sub"
#define COMPONENT_BUF "Buffer"
#define COMPONENT_TEHB "TEHB"
#define COMPONENT_OEHB "OEHB"
#define COMPONENT_FIFO "Fifo"
#define COMPONENT_NFIFO "nFifo"
#define COMPONENT_TFIFO "tFifo"
#define COMPONENT_FORK "Fork"
#define COMPONENT_LFORK "LazyFork"
#define COMPONENT_ICMP "icmp"
#define COMPONENT_CONSTANT_ "Constant"
#define COMPONENT_BRANCH "Branch"
#define COMPONENT_END "Exit"
#define COMPONENT_START "Entry"
#define COMPONENT_SOURCE "Source"
#define COMPONENT_SINK "Sink"
#define COMPONENT_MUX "Mux"
#define COMPONENT_CTRLMERGE "CntrlMerge"
#define COMPONENT_LSQ "LSQ"
#define COMPONENT_MC "MC"
#define COMPONENT_SEL "SEL"
#define COMPONENT_DISTRIBUTOR "Distributor"
#define COMPONENT_SELECTOR "Selector"
#define COMPONENT_INJECTOR "Inj"
#define COMPONENT_SPEC_BRANCH "speculating_branch"
#define COMPONENT_COMMIT_UNIT "spec_commit"           // "commit_unit"
#define COMPONENT_SAVE_COMMIT_UNIT "spec_save_commit" // "save_commit_unit"
#define COMPONENT_SAVE_UNIT "spec_save"               // "save_unit"
#define COMPONENT_SPECULATOR "speculator"

#define UNDERSCORE "_"
#define COLOUMN " : "
#define SEMICOLOUMN ";"
#define COMMA ","

#define SIGNAL_STRING "signal "
#define STD_LOGIC "std_logic;"

#define DEFAULT_BITWIDTH 32

enum {
  ENTITY_MERGE_INDX,
  ENTITY_READ_MEMORY_INDX,
  ENTITY_SINGLE_OP_INDX,
  ENTITY_GET_PTR_INDX,
  ENTITY_INT_MUL_INDX,
  ENTITY_INT_ADD_INDX,
  ENTITY_INT_SUB_INDX,
  ENTITY_BUF_INDX,
  ENTITY_TEHB_INDX,
  ENTITY_OEHB_INDX,
  ENTITY_FIFO_INDX,
  ENTITY_NFIFO_INDX,
  ENTITY_TFIFO_INDX,
  ENTITY_FORK_INDX,
  ENTITY_LFORK_INDX,
  ENTITY_ICMP_INDX,
  ENTITY_CONSTANT_INDX,
  ENTITY_BRANCH_INDX,
  ENTITY_END_INDX,
  ENTITY_START_INDX,
  ENTITY_WRITE_MEMORY_INDX,
  ENTITY_SOURCE_INDX,
  ENTITY_SINK_INDX,
  ENTITY_MUX_INDX,
  ENTITY_CTRLMERGE_INDX,
  ENTITY_LSQ_INDX,
  ENTITY_MC_INDX,
  ENTITY_SEL_INDX,
  ENTITY_DISTRIBUTOR_INDX,
  ENTITY_SELECTOR_INDX,
  ENTITY_SPEC_BRANCH_INDX,
  ENTITY_COMMIT_UNIT_INDX,
  ENTITY_SAVE_COMMIT_UNIT_INDX,
  ENTITY_SAVE_UNIT_INDX,
  ENTITY_SPECULATOR_INDX,
  ENTITY_MAX
};

extern bool ENABLE_SPECULATION;

void writeVHDL(const std::string &kernelName, const std::string &outFolder);

#endif // EXPORT_VHDL_VHDL_WRITER_H
