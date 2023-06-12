//===- UtilsForPlaceBuffers.cpp - functions for placing buffer  -*- C++ -*-===//
//
// This file implements function supports for buffer placement.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/UtilsForPlaceBuffers.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/IndentedOstream.h"
#include <optional>

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;

bool buffer::isEntryOp(Operation *op,
                       std::vector<Operation *> &visitedOp) {
    if (op->getAttrs().data()->getName() == "bb") {
      if (op->getAttrOfType<IntegerAttr>("bb").getUInt() == 0 &&
          isa<MergeOp>(op)) {
        // visitedOp.push_back(op);
        return true;
      }
      
  }
  return false;
}

int buffer::getBBIndex(Operation *op) {
  if (op->getAttrs().data()->getName() == "bb")
    return op->getAttrOfType<IntegerAttr>("bb").getUInt();
  return -1;
}

bool buffer::isBackEdge(Operation *opSrc, Operation *opDst) {
  // unsigned bbSrcInd = getBBIndex(opSrc);
  // unsigned bbDstInd = getBBIndex(opDst);
  // if (bbSrcInd == UINT_MAX || bbDstInd == UINT_MAX ||
  //     isa<handshake::EndOp>(*opDst))
  //   return false;

  // if (bbSrcInd > bbDstInd)
  //   return true;

  if (opDst->isProperAncestor(opSrc))
    return true;

  return false;
}


unit *buffer::getUnitWithOp(Operation *op, std::vector<unit *> &unitList) {
  for (auto u : unitList) {
    if (u->op == op)
      return u;
  }
  return nullptr;
}

void buffer::connectInChannel(unit *unitNode, channel *inChannel) {
    port *inPort = new port(inChannel->valPort);
    inPort->cntChannels.push_back(inChannel);
    unitNode->inPorts.push_back(inPort);
}

void buffer::dfsHandshakeGraph(Operation *opNode, 
                               std::vector<unit *> &unitList,
                               std::vector<Operation *> &visited,
                               channel *inChannel) {

  // ensure inChannel is marked as connected to the unit
  if (inChannel != nullptr)
    if (unit *unitNode = getUnitWithOp(opNode, unitList);
        unitNode != nullptr) {
      connectInChannel(unitNode, inChannel);
    }

  if (std::find(visited.begin(), visited.end(), opNode) != visited.end()) {
    return;
  }

  // marked as visited
  visited.push_back(opNode);

  // initialize the unit node 
  unit *unitNode = new unit(opNode);
  unitList.push_back(unitNode);
  if (inChannel != nullptr)
    connectInChannel(unitNode, inChannel);

  // dfs the successor operation
  for (auto resOperand : opNode->getResults()) {
    // initialize the out port
    port *outPort = new port(&resOperand);
    for (auto sucOp : resOperand.getUsers()) {
      // create the channel connected to the outport
      channel *outChannel = new channel(opNode, sucOp, &resOperand);
      outChannel->isBackEdge = isBackEdge(opNode, sucOp);
      // llvm::errs() << "creating channels .... \n";
      // outChannel->print();
      outPort->cntChannels.push_back(outChannel);

      // dfs the successor operation
      dfsHandshakeGraph(sucOp, unitList, visited, outChannel);
    } 
    unitNode->outPorts.push_back(outPort);
  }
  

}


// bool buffer::isConnected(basicBlock *bb, Operation *op) {
//   for (auto channel : bb->outChannels) {
//     if (auto connectOp = (channel->opDst).value(); connectOp == op) {
//       return true;
//     }
//   }
//   return false;
// }

// arch *buffer::findExistsArch(basicBlock *bbSrc, basicBlock *bbDst,
//                              std::vector<arch *> &archList) {
//   for (auto arch : archList)
//     if (arch->bbSrc == bbSrc && arch->bbDst == bbDst)
//       return arch;
//   return nullptr;
// }

// basicBlock *buffer::findExistsBB(unsigned bbInd,
//                                  std::vector<basicBlock *> &bbList) {
//   for (auto bb : bbList)
//     if (bb->index == bbInd)
//       return bb;
//   return nullptr;
// }

// // link the basic blocks via channel between operations
// void buffer::linkBBViaChannel(Operation *opSrc, Operation *opDst,
//                               unsigned newbbInd, basicBlock *curBB,
//                               std::vector<basicBlock *> &bbList) {
//   auto *outChannel = new channel();
//   outChannel->opSrc = opSrc;
//   outChannel->opDst = opDst;
//   // outChannel->isOutEdge = true;
//   outChannel->bbSrc = curBB;
//   if (auto bbDst = findExistsBB(newbbInd, bbList); bbDst != nullptr) {
//     outChannel->bbDst = bbDst;
//     if (bbDst->index <= curBB->index)
//       outChannel->isBackEdge = true;
//   } else {
//     basicBlock *bb = new basicBlock();
//     bb->index = newbbInd;
//     outChannel->bbDst = bb;
//     bbList.push_back(bb);
//   }
//   curBB->outChannels.push_back(outChannel);
//   return;
// }

// void buffer::dfsBBGraphs(Operation *opNode, std::vector<Operation *> &visited,
//                          basicBlock *curBB, std::vector<basicBlock *> &bbList) {
//   if (std::find(visited.begin(), visited.end(), opNode) != visited.end()) {
//     return;
//   }
//   // marked as visited
//   visited.push_back(opNode);

//   if (isa<handshake::EndOp>(*opNode)) {
//     curBB->isExitBB = true;
//     return;
//   }

//   // vectors to store successor operation
//   SmallVector<Operation *> sucOps;

//   for (auto sucOp : opNode->getResults().getUsers()) {
//     // llvm::errs() << "dfs graph " << *sucOp << "\n";
//     // get the index of the successor basic block
//     unsigned bbInd = getBBIndex(sucOp);

//     // not in a basic block
//     if (bbInd == UINT_MAX)
//       continue;

//     if (bbInd != getBBIndex(opNode)) {
//       // llvm::errs() << "linkBBVia " << *sucOp << "\n";
//       // if not in the same basic block, link via out arc
//       linkBBViaChannel(opNode, sucOp, bbInd, curBB, bbList);
//       // stop tranversing nodes not in the same basic block
//       continue;
//     } else if (isBackEdge(opNode, sucOp)) {
//       // llvm::errs() << "linkBBVia (backedge) " << *sucOp << "\n";
//       // need to determine whether is a back edge in a same block
//       linkBBViaChannel(opNode, sucOp, bbInd, curBB, bbList);
//     }

//     dfsBBGraphs(sucOp, visited, curBB, bbList);
//   }
//   // llvm::errs() << "dfs bb " << getBBIndex(opNode) << "\n";
// }



/// ================== dataFlowCircuit Function ================== ///
void buffer::dataFlowCircuit::printCircuits() {
  for (auto unit : units) {
    llvm::errs() << "===========================\n";
    llvm::errs() << "operation: " << *(unit->op) << "\n";
    llvm::errs() << "-------------inPorts: \n";
    for (auto port : unit->outPorts)
      for (auto ch : port->cntChannels)
        ch->print();
    llvm::errs() << "-------------outPorts: \n";
    for (auto port : unit->inPorts)
      for (auto ch : port->cntChannels)
        ch->print();
  }
}

// std::vector<std::vector<float>>
// buffer::dataFlowCircuit::readInfoFromFile(const std::string &filename) {
//   std::vector<std::vector<float>> info;

//   std::ifstream file(filename);
//   assert(file.is_open() && "Error opening delay info file");

//   std::string line;
//   while (std::getline(file, line)) {
//     std::vector<float> row;
//     std::istringstream iss(line);
//     std::string value;

//     while (std::getline(iss, value, ',')) {
//       float num = std::stof(value);
//       row.push_back(num);
//     }

//     assert(!row.empty() && "Error reading delay info file");
//     info.push_back(row);
//   }

//   file.close();

//   return info;
// }

// void buffer::dataFlowCircuit::initUnitTimeInfo() {
//   for (auto u : units) {
//     Operation *op = u->op;
//     std::string opName = op->getName().getStringRef().str();
//     // ignore handshake name 
//     // for example, "handshake.merge" -> "merge"
//     size_t dotPos = opName.find_last_of(".");
//     llvm::errs() << "opName: " << opName.substr(dotPos + 1) << "\n";
    

//     if (compNameToIndex.count(opName.substr(dotPos + 1)) == 0) 
//       continue;
    
//     int compInd = compNameToIndex[opName.substr(dotPos + 1)];
//     int bitInd = bitWidthToIndex[32];

//     Type resType = op->getResult(0).getType();
//     if (isa<IntegerType>(resType) &&
//         bitWidthToIndex.count(resType.getIntOrFloatBitWidth()) > 0)
//       bitInd = bitWidthToIndex[resType.getIntOrFloatBitWidth()];
//     u->delay = delayInfo[compInd][bitInd];
//     u->latency = latencyInfo[compInd][bitInd];
//     llvm::errs() << *op << " delay: " << u->delay << "\n";
//     llvm::errs() << *op << " latency: " << u->latency << "\n";
//   }
// }
// void buffer::dataFlowCircuit::insertSelBB(handshake::FuncOp funcOp,
//                                           basicBlock *bb) {

//   selBBs.push_back(bb);

//   unsigned bbInd = bb->index;
//   llvm::errs() << "==================================\n";
//   llvm::errs() << "create channels in BB " << bbInd << "\n";

//   for (auto &op : funcOp.getOps()) {
//     // llvm::errs() << "------------op: " << op << "\n";
//     if (getBBIndex(&op) != bbInd)
//       continue;
//     // insert channels if op and its successor are in the same basic block

//     // create units w.r.t op
//     if (!hasUnit(&op)) {
//       unit *unitInfo = new unit();
//       unitInfo->op = &op;
//       // units.insert(std::make_pair(&op, unitInfo));
//       units.push_back(unitInfo);
//     }

//     // llvm::errs() << "Try to create channels inner BB \n";
//     for (auto sucOp : op.getResults().getUsers()) {
//       if (!hasUnit(sucOp)) {
//         unit *sucUnitInfo = new unit();
//         sucUnitInfo->op = sucOp;
//         units.push_back(sucUnitInfo);
//       }

//       // ignore backedges
//       if (isBackEdge(&op, sucOp))
//         continue;

//       // create channels
//       if (getBBIndex(sucOp) == bbInd && !hasChannel(&op, sucOp)) {
//         // llvm::errs() << "creating channels inner BB " << bbInd << "\n";
//         auto *innerCh = new channel();
//         innerCh->opSrc = &op;
//         innerCh->opDst = sucOp;
//         innerCh->bbSrc = bb;
//         innerCh->bbDst = bb;
//         // innerCh->print();
//         channels.push_back(innerCh);
//         units[findUnitIndex(&op)]->outChannels.push_back(innerCh);
//         units[findUnitIndex(sucOp)]->inChannels.push_back(innerCh);
//       }
//     }
//   }
// }

// void buffer::dataFlowCircuit::insertSelArc(arch *arc) {
//   // find channels in vector: outChannels w.r.t to the arch
//   basicBlock *bbSrc = arc->bbSrc;
//   basicBlock *bbDst = arc->bbDst;

//   llvm::errs() << "==================================\n";
//   llvm::errs() << "create channels from BB " << arc->bbSrc->index << " to "
//                << arc->bbDst->index << "\n";

//   for (channel *ch : bbSrc->outChannels)
//     if (ch->bbDst == bbDst) {
//       insertChannel(ch);
//       // ch->print();
//       Operation *opSrc = (ch->opSrc).value();
//       Operation *opDst = (ch->opDst).value();
//       assert(findUnitIndex(opSrc) != -1 && "Invalid Operation");
//       units[findUnitIndex(opSrc)]->outChannels.push_back(ch);

//       assert(findUnitIndex(opDst) != -1 && "Invalid Operation");
//       units[findUnitIndex(opDst)]->inChannels.push_back(ch);
//     }
// }

// LogicalResult buffer::dataFlowCircuit::initMLIPModelVars(
//     GRBModel &milpModel,
//     GRBVar &varThrpt,
//     std::vector<std::map<std::string, GRBVar>> &channelVars,
//     std::vector<std::map<std::string, GRBVar>> &unitVars) {

//   initUnitTimeInfo();

//   // init variables of the models
//   varThrpt = milpModel.addVar(0, 1, 0.0, GRB_CONTINUOUS, "thrpt");
//   for (int i = 0; i < channels.size(); i++) {
//     std::map<std::string, GRBVar> chVar;
//     chVar["R_c" + std::to_string(i)] =
//         milpModel.addVar(0, 1, 0.0, GRB_BINARY, "R_c" + std::to_string(i));
//     chVar["N_c" + std::to_string(i)] = milpModel.addVar(
//         0, UINT32_MAX, 0.0, GRB_INTEGER, "N_c" + std::to_string(i));

//     std::string tInChannel = "t_c" + std::to_string(i);
//     chVar[tInChannel + "_in"] = milpModel.addVar(
//         0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, tInChannel + "_in");
//     chVar[tInChannel + "_out"] = milpModel.addVar(
//         0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, tInChannel + "_out");

//     chVar["thrpt_tok_c" + std::to_string(i)] = milpModel.addVar(
//         0, 1, 0.0, GRB_CONTINUOUS, "thrpt_tok_c" + std::to_string(i));
//     chVar["thrpt_emp_c" + std::to_string(i)] = milpModel.addVar(
//         0, 1, 0.0, GRB_CONTINUOUS, "thrpt_emp_c" + std::to_string(i));
//     channelVars.push_back(chVar);
//   }

//   int i = 0;
//   for (auto unit : units) {
//     std::map<std::string, GRBVar> unitVar;
//     Operation *op = unit->op;
//     unitVar["r_u" + std::to_string(i)] =
//         milpModel.addVar(0, 1, 0.0, GRB_CONTINUOUS, "r_u" + std::to_string(i));
//     unit->ind = i++;
//     unitVars.push_back(unitVar);
//   }

//   return success();
// }

// LogicalResult buffer::dataFlowCircuit::createMILPPathConstrs(
//     GRBModel &milpModel,
//     std::vector<std::map<std::string, GRBVar>> &channelVars,
//     std::vector<std::map<std::string, GRBVar>> &unitVars) {

//   for (int i=0; i<channelVars.size(); i++) {
//     std::string tInChannel = "t_c" + std::to_string(i) + "_in";
//     std::string tOutChannel = "t_c" + std::to_string(i) + "_out";
//     // milpModel.addConstr(channelVars[i][tOutChannel]
//     // t_c^{out} >= t_c^{in} + P_{max} R_c
//     milpModel.addConstr(channelVars[i][tOutChannel] >=
//                         channelVars[i][tInChannel] -
//                         maxCP *
//                         channelVars[i]["R_c" + std::to_string(i)]);
//   }

 
//   for (int i=0; i<unitVars.size(); i++) {
//     std::string rUnit = "r_u" + std::to_string(i);
//     unit *u = units[i];
//     double delay = u->delay;

//      // P >= t_{c2}^{in} & t_{c2}^{in} >= t_{c1}^{out} + D_u
//     for (auto ch1 : u->inChannels) {
//       size_t ch1Ind = findChannelIndex(ch1);
//       std::string tOutChannel = "t_c" + std::to_string(ch1Ind) + "_out";
//       for (auto ch2 : u->outChannels) {
//         size_t ch2Ind = findChannelIndex(ch2);
//         std::string tInChannel = "t_c" + std::to_string(ch2Ind) + "_in";
//         // P >= t_{c2}^{in} 
//         milpModel.addConstr(targetCP >=
//                             channelVars[ch2Ind][tInChannel]);
//         // t_{c2}^{in} >= t_{c1}^{out} + D_u
//         milpModel.addConstr(channelVars[ch2Ind][tInChannel] >=
//                             channelVars[ch1Ind][tOutChannel] + delay);
//       }
//     }
//     // llvm::errs() << "=========path constrs=========\n";
//     // llvm::errs() << "unit: " << *(u->op) << "\n";
//     // llvm::errs() << "inChannels: \n";
//     // for (auto ch : u->inChannels) {
//     //   ch->print();
//     // }
//     // llvm::errs() << "outChannels: \n";
//     //  for (auto ch : u->outChannels) {
//     //   ch->print();
//     // }
//   }

//   return success();
// }

// LogicalResult buffer::dataFlowCircuit::createMILPThroughputConstrs(
//     GRBModel &milpModel,
//     GRBVar &varThrpt,
//     std::vector<std::map<std::string, GRBVar>> &channelVars,
//     std::vector<std::map<std::string, GRBVar>> &unitVars) {

//   // GRBVar throughput = milpModel.getVarByName("thrpt");

//   for (int i=0; i<channelVars.size(); i++) {
//     std::string thrptTok = "thrpt_tok_c" + std::to_string(i);
//     int isbackedge = channels[i]->isBackEdge ? 1 : 0;

//     unit *uSrc = units[findUnitIndex((channels[i]->opSrc).value())];
//     unit *uDst = units[findUnitIndex((channels[i]->opSrc).value())];

//     int uSrcInd = findUnitIndex((channels[i]->opSrc).value());
//     int uDstInd = findUnitIndex((channels[i]->opDst).value());
//     // \dot{throughput}_c = R_c + r_u^{src} - r_u^{dst}
//     milpModel.addConstr(channelVars[i][thrptTok] == 
//                         isbackedge + 
//                         unitVars[uSrcInd]["r_u" + std::to_string(uSrcInd)] -
//                         unitVars[uDstInd]["r_u" + std::to_string(uDstInd)]);
//     // throughput <= \dot{throughput}_c - R_c + 1
//     milpModel.addConstr(varThrpt <=
//                         channelVars[i][thrptTok] -
//                         channelVars[i]["R_c" + std::to_string(i)] + 1);

//     milpModel.addConstr(channelVars[i]["N_c"+std::to_string(i)] >=
//                         channelVars[i]["R_c" + std::to_string(i)] + 1);
//   }    

//   return success();
// }

// LogicalResult buffer::dataFlowCircuit::defineCostFunction(
//     GRBModel &milpModel,
//     GRBVar &varThrpt,
//     std::vector<std::map<std::string, GRBVar>> &channelVars,
//     std::vector<std::map<std::string, GRBVar>> &unitVars) {

//   GRBLinExpr objExpr = varThrpt; 

//   double lumbdaCoef = 1e-7;
//   for (int i=0; i<channelVars.size(); i++) {
//     objExpr -= lumbdaCoef * channelVars[i]["N_c" + std::to_string(i)];
//   }

//   milpModel.setObjective(objExpr, GRB_MAXIMIZE);
//   return success();

// }

// void buffer::dataFlowCircuit::insertBuffersInChannel
//   (MLIRContext *ctx, channel *ch, bool fifo, int slots) {
//   OpBuilder builder(ctx);

//   Operation *opSrc = ch->opSrc.value();
//   Operation *opDst = ch->opDst.value();

//   builder.setInsertionPointAfter(opSrc);
//   auto bufferOp = builder.create<handshake::BufferOp>
//                     (opSrc->getLoc(),
//                      opSrc->getResult(0).getType(),
//                      opSrc->getResult(0));

//   if (fifo)
//     bufferOp.setBufferType(BufferTypeEnum::fifo);
//   else
//     bufferOp.setBufferType(BufferTypeEnum::seq);
//   bufferOp.setSlots(slots);
//   opSrc->getResult(0).replaceUsesWithIf(bufferOp.getResult(),
//                                         [&](OpOperand &operand) {
//                                           // return true;
//                                           return operand.getOwner() == opDst;
//                                         });
//   // llvm::errs() << "Inserted " << slots << " buffer: ";
// }

// LogicalResult buffer::dataFlowCircuit::instantiateBuffers(
//   MLIRContext *ctx,
//   GRBModel &milpModel,
//   GRBVar &varThrpt,
//   std::vector<std::map<std::string, GRBVar>> &channelVars,
//   std::vector<std::map<std::string, GRBVar>> &unitVars) {

//   double x = varThrpt.get(GRB_DoubleAttr_X);
//   llvm::errs() << "Thourghput: " << varThrpt.get(GRB_DoubleAttr_X) << "\n";
//   // varThrpt.get(GRB_DoubleAttr_X);

//   for (int i=0; i<channelVars.size(); i++) {
//     auto N_c = channelVars[i]["N_c" + std::to_string(i)].get(GRB_DoubleAttr_X);
//     int N_c_int = (int) N_c;
//     auto R_c = channelVars[i]["R_c" + std::to_string(i)].get(GRB_DoubleAttr_X);
//     if (N_c_int > 0 && R_c>0) 
//       insertBuffersInChannel(ctx, channels[i], true, N_c_int-1);

//     if (N_c_int > 1 && R_c ==0) {
//       insertBuffersInChannel(ctx, channels[i], false, N_c_int-1);
//     }

//     // if (R_c > 0) 
//     //     llvm::errs() << " : sequential\n";
//     // else 
//     //     llvm::errs() << " : transparent\n";
  
//   }

//     return success();
// }