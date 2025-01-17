//===- SubjectGraph.h - Exp. support for MAPBUF buffer placement -------*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements Subject Graph constructors.
//
//===-----------------------------------------------------------------===//

#ifndef EXPERIMENTAL_SUPPORT_SUBJECT_GRAPH_H
#define EXPERIMENTAL_SUPPORT_SUBJECT_GRAPH_H

#include "BlifReader.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Conversion/HandshakeToHW.h"
#include "dynamatic/Dialect/HW/HWOpInterfaces.h"
#include "dynamatic/Dialect/HW/HWOps.h"
#include "dynamatic/Dialect/HW/HWTypes.h"
#include "dynamatic/Dialect/HW/PortImplementation.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Utils/Utils.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <boost/functional/hash/extensions.hpp>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace mlir;

namespace dynamatic {
namespace experimental {

struct ChannelSignals {
  std::vector<Node *> dataSignals;
  Node *validSignal;
  Node *readySignal;
};

class BaseSubjectGraph {
public:
  static inline DenseMap<Operation *, BaseSubjectGraph *> moduleMap;
  static inline DenseMap<BaseSubjectGraph *, Operation *> subjectGraphMap;
  static inline std::string baseBlifPath;

protected:
  bool isBlackbox = false;
  Operation *op;
  std::vector<Operation *> inputModules;
  std::vector<Operation *> outputModules;
  DenseMap<Operation *, unsigned int> inputModuleToResNum;
  DenseMap<Operation *, unsigned int> outputModuleToResNum;

  std::vector<BaseSubjectGraph *> inputSubjectGraphs;
  std::vector<BaseSubjectGraph *> outputSubjectGraphs;
  DenseMap<BaseSubjectGraph *, unsigned int> inputSubjectGraphToResNum;
  DenseMap<BaseSubjectGraph *, unsigned int> outputSubjectGraphToResNum;

  std::string fullPath;
  std::string moduleType;
  std::string uniqueName;
  BlifData *blifData;

  void assignSignals(ChannelSignals &signals, Node *node,
                     const std::string &nodeName);
  void connectInputNodesHelper(ChannelSignals &currentSignals,
                               BaseSubjectGraph *moduleBeforeSubjectGraph);

public:
  BaseSubjectGraph();
  BaseSubjectGraph(Operation *op);
  BaseSubjectGraph(Operation *before, Operation *after);

  void replaceOpsBySubjectGraph();
  static unsigned int getChannelNumber(BaseSubjectGraph *first,
                                       BaseSubjectGraph *second);
  static void changeOutput(BaseSubjectGraph *graph, BaseSubjectGraph *newOutput,
                           BaseSubjectGraph *oldOutput);
  static void changeInput(BaseSubjectGraph *graph, BaseSubjectGraph *newInput,
                          BaseSubjectGraph *oldInput);
  void appendVarsToPath(std::initializer_list<unsigned int> inputs);
  void connectSignals(Node *currentSignal, Node *beforeSignal);
  BlifData *getBlifData() const;
  static BaseSubjectGraph *getSubjectGraph(Operation *op);
  std::string &getUniqueNameGraph();
  std::string &getModuleType();
  std::vector<BaseSubjectGraph *> getInputSubjectGraphs();
  std::vector<BaseSubjectGraph *> getOutputSubjectGraphs();
  static void setBaseBlifPath(llvm::StringRef path) {
    baseBlifPath = path.str();
  }

  virtual ~BaseSubjectGraph() = default;
  virtual void connectInputNodes() = 0;
  virtual ChannelSignals &returnOutputNodes(unsigned int) = 0;
};

class ArithSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int dataWidth = 0;
  std::unordered_map<unsigned int, ChannelSignals> inputNodes;
  ChannelSignals outputNodes;

public:
  ArithSubjectGraph(Operation *op);
  void connectInputNodes() override;
  ChannelSignals &returnOutputNodes(unsigned int channelIndex) override;
};

class CmpISubjectGraph : public BaseSubjectGraph {
private:
  unsigned int dataWidth = 0;
  std::unordered_map<unsigned int, ChannelSignals> inputNodes;
  ChannelSignals outputNodes;

public:
  CmpISubjectGraph(Operation *op);
  void connectInputNodes() override;
  ChannelSignals &returnOutputNodes(unsigned int channelIndex) override;
};

class ForkSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int size = 0;
  unsigned int dataWidth = 0;
  std::unordered_map<unsigned int, ChannelSignals> outputNodes;
  ChannelSignals inputNodes;

public:
  ForkSubjectGraph(Operation *op);
  void connectInputNodes() override;
  ChannelSignals &returnOutputNodes(unsigned int channelIndex) override;
};

class MuxSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int size = 0;
  unsigned int dataWidth = 0;
  unsigned int selectType = 0;
  std::unordered_map<unsigned int, ChannelSignals> inputNodes;
  ChannelSignals indexNodes;
  ChannelSignals outputNodes;

public:
  MuxSubjectGraph(Operation *op);
  void connectInputNodes() override;
  ChannelSignals &returnOutputNodes(unsigned int channelIndex) override;
};

class ControlMergeSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int size = 0;
  unsigned int dataWidth = 0;
  unsigned int indexType = 0;
  std::unordered_map<unsigned int, ChannelSignals> inputNodes;
  ChannelSignals indexNodes;
  ChannelSignals outputNodes;

public:
  ControlMergeSubjectGraph(Operation *op);
  void connectInputNodes() override;
  ChannelSignals &returnOutputNodes(unsigned int channelIndex) override;
};

class ConditionalBranchSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int dataWidth = 0;
  ChannelSignals conditionNodes;
  ChannelSignals inputNodes;
  std::unordered_map<unsigned int, ChannelSignals> outputNodes;

public:
  ConditionalBranchSubjectGraph(Operation *op);
  void connectInputNodes() override;
  ChannelSignals &returnOutputNodes(unsigned int channelIndex) override;
};

class SourceSubjectGraph : public BaseSubjectGraph {
private:
  ChannelSignals outputNodes;

public:
  SourceSubjectGraph(Operation *op);
  void connectInputNodes() override;
  ChannelSignals &returnOutputNodes(unsigned int) override;
};

class LoadSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int dataWidth = 0;
  unsigned int addrType = 0;
  ChannelSignals addrInSignals;
  ChannelSignals addrOutSignals;
  ChannelSignals dataOutSignals;

public:
  LoadSubjectGraph(Operation *op);
  void connectInputNodes() override;
  ChannelSignals &returnOutputNodes(unsigned int channelIndex) override;
};

class StoreSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int dataWidth = 0;
  unsigned int addrType = 0;
  ChannelSignals dataInSignals;
  ChannelSignals addrInSignals;
  ChannelSignals addrOutSignals;

public:
  StoreSubjectGraph(Operation *op);
  void connectInputNodes() override;
  ChannelSignals &returnOutputNodes(unsigned int channelIndex) override;
};

class ConstantSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int dataWidth = 0;
  ChannelSignals controlSignals;
  ChannelSignals outputNodes;

public:
  ConstantSubjectGraph(Operation *op);
  void connectInputNodes() override;
  ChannelSignals &returnOutputNodes(unsigned int channelIndex) override;
};

class ExtTruncSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int inputWidth = 0;
  unsigned int outputWidth = 0;
  ChannelSignals inputNodes;
  ChannelSignals outputNodes;

public:
  ExtTruncSubjectGraph(Operation *op);
  void connectInputNodes() override;
  ChannelSignals &returnOutputNodes(unsigned int channelIndex) override;
};

class SelectSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int dataWidth = 0;
  std::unordered_map<unsigned int, ChannelSignals> inputNodes;
  ChannelSignals outputNodes;

public:
  SelectSubjectGraph(Operation *op);
  void connectInputNodes() override;
  ChannelSignals &returnOutputNodes(unsigned int channelIndex) override;
};

class MergeSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int dataWidth = 0;
  unsigned int size = 0;
  std::unordered_map<unsigned int, ChannelSignals> inputNodes;
  ChannelSignals outputNodes;

public:
  MergeSubjectGraph(Operation *op);
  void connectInputNodes() override;
  ChannelSignals &returnOutputNodes(unsigned int channelIndex) override;
};

class BranchSinkSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int dataWidth = 0;
  ChannelSignals inputNodes;
  ChannelSignals outputNodes;

public:
  BranchSinkSubjectGraph(Operation *op);
  void connectInputNodes() override;
  ChannelSignals &returnOutputNodes(unsigned int) override;
};

enum class BufferType { OEHB, TEHB };
class BufferSubjectGraph;

struct BufferPair {
  BufferSubjectGraph *oehb;
  BufferSubjectGraph *tehb;

  BufferPair() : oehb(nullptr), tehb(nullptr) {}
  BufferPair(BufferSubjectGraph *o, BufferSubjectGraph *t) : oehb(o), tehb(t) {}
};

class BufferSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int dataWidth = 0;
  ChannelSignals inputNodes;
  ChannelSignals outputNodes;

  static std::string getBufferTypeName(BufferType type) {
    switch (type) {
    case BufferType::OEHB:
      return "oehb";
    case BufferType::TEHB:
      return "tehb";
    }
  }

public:
  BufferSubjectGraph(Operation *op);
  BufferSubjectGraph(Operation *op1, Operation *op2, std::string bufferType);
  BufferSubjectGraph(BufferSubjectGraph *graph1, Operation *op2,
                     std::string bufferType);
  void connectInputNodes() override;
  ChannelSignals &returnOutputNodes(unsigned int) override;

  static BufferPair createBuffers(Operation *inputOp, Operation *outputOp) {
    // Create OEHB buffer using the temporary string
    BufferSubjectGraph *oehb = new BufferSubjectGraph(
        inputOp, outputOp, getBufferTypeName(BufferType::OEHB));

    // Create TEHB buffer that connects to OEHB
    BufferSubjectGraph *tehb = new BufferSubjectGraph(
        oehb, outputOp, getBufferTypeName(BufferType::TEHB));

    return BufferPair(oehb, tehb);
  }
};

class OperationDifferentiator {
  Operation *op;

public:
  static inline DenseMap<Operation *, BaseSubjectGraph *> moduleMap;
  OperationDifferentiator(Operation *ops);
};

class SubjectGraphGenerator {
public:
  SubjectGraphGenerator(handshake::FuncOp funcOp, StringRef blifFiles);
};

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_SUBJECT_GRAPH_H