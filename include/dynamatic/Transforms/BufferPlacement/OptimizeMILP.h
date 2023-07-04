//===- OptimizeMILP.h - optimize MILP model over CFDFC  ---------*- C++ -*-===//
//
// This file declaresfunction the functions of MILP solver for buffer placement.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_OPTIMIZEMILP_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_OPTIMIZEMILP_H

#include "dynamatic/Support/BufferingStrategy.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/BufferPlacement/ExtractMG.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

namespace dynamatic {
namespace buffer {

using namespace circt;
using namespace handshake;

inline Operation *getUserOp(Value val) {
  auto dstOp = val.getUsers().begin();
  llvm::errs() << "first user: " << *dstOp << "\n";
  unsigned numUsers = 0;
  for (auto c : val.getUsers()) {
    numUsers++;
  }
  llvm::errs() << "user number: " << numUsers << "\n";
  // llvm::errs() << "last user:" << *(val.getUses()) << "\n";
  if (!isa<handshake::ForkOp>(*dstOp))
    assert(dstOp == val.getUsers().end() && "There are multiple users!");
  return *dstOp;
}

struct UnitVar {
public:
  GRBVar retIn, retOut;
};

struct ChannelVar {
public:
  GRBVar tDataIn, tDataOut, tElasIn, tElasOut;
  GRBVar tValidIn, tValidOut, tReadyIn, tReadyOut;
  GRBVar thrptTok, bufIsOp, bufNSlots, hasBuf;
  GRBVar valbufIsOp, readybufIsOp;
};

struct Result {
  bool opaque;
  unsigned numSlots;
};

LogicalResult placeBufferInCFDFCircuit(CFDFC &CFDFCircuit,
                                       std::map<Value *, Result> &res);

// struct DataflowCircuit {
//   // for read delay file
//   std::map<std::string, int> compNameToIndex = {
//       {"cmpi", 0},
//       {"addi", 1},
//       {"subi", 2},
//       {"muli", 3},
//       {"extsi", 4},
//       // {"load", 5}, {"store", 6},  ????
//       {"d_load", 5},
//       {"d_store", 6},
//       {"LsqLoad", 7},  // ?
//       {"LsqStore", 8}, // ?
//       // {"merge", 9},    // ??
//       {"Getelementptr", 10},
//       {"Addf", 11},
//       {"Subf", 12},
//       {"Mulf", 13},
//       {"divu", 14},
//       {"Divs", 15},
//       {"Divf", 16},
//       {"cmpf", 17},
//       // {"Phic", 18}, // ---> merge & control merge
//       {"merge", 18},
//       {"control_merge", 18},
//       {"zdl", 19},
//       {"fork", 20},
//       {"Ret", 21}, // handshake.return ?
//       {"br", 22},  // conditional branch ??
//       {"cond_br", 22},
//       {"end", 23},
//       {"and", 24},
//       {"or", 25},
//       {"xori", 26},
//       {"Shl", 27},
//       {"Ashr", 28},
//       {"Lshr", 29},
//       {"select", 30},
//       {"mux", 31}};

//   std::map<int, int> bitWidthToIndex = {{1, 0},  {2, 1},  {4, 2}, {8, 3},
//                                         {16, 4}, {32, 5}, {64, 6}};

//   std::string infoFielDefault = std::getenv("LEGACY_DYNAMATICPP");
//   std::string delayFile = infoFielDefault +
//   "/data/targets/default_delay.dat"; std::string latencyFile =
//       infoFielDefault + "data/targets/default_latency.dat";

//   std::vector<std::vector<float>> delayInfo;
//   std::vector<std::vector<float>> latencyInfo;

//   double targetCP, maxCP;
//   std::vector<Operation *> units;
//   std::vector<Value> channels;

//   int execN = 0;

//   void printCircuits();

//   std::vector<std::vector<float>> readInfoFromFile(const std::string
//   &filename);

//   LogicalResult createMILPModel(dynamatic::ChannelBufProps &strategy,
//                                 std::map<Value *, Result> &res);

//   double getTimeInfo(Operation *unit, std::string infoName);

//   LogicalResult
//   createModelConstraints(dynamatic::ChannelBufProps &strategy,
//                          GRBModel &modelBuf, GRBVar &thrpt,
//                          std::map<Operation *, UnitVar> &unitVars,
//                          std::map<Value *, ChannelVar> &channelVars);
// };
} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_OPTIMIZEMILP_H