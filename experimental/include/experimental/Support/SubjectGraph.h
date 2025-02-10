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
#include "dynamatic/Dialect/HW/PortImplementation.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <boost/functional/hash/extensions.hpp>
#include <string>
#include <unordered_map>
#include <vector>

using namespace mlir;

namespace dynamatic {
namespace experimental {

/// Struct that holds the different types of signals that a channel can have
struct ChannelSignals {
  std::vector<Node *> dataSignals;
  Node *validSignal;
  Node *readySignal;
};

/// Base class for all subject graphs. This class represents the subject graph
/// of an Operation in MLIR.
/// Each BaseSubjectGraph maintains information about:
/// - Input/output connections to other modules and subject graphs
/// - Signal routing for handshake protocols (data, valid, ready signals)
/// - Path information for BLIF (Berkeley Logic Interchange Format) file
/// generation
/// - Unique naming and module type identification
class BaseSubjectGraph {
protected:
  // Operation that the SubjectGraph is based on
  Operation *op;

  // moduleType is used to generate the path to the BLIF file
  std::string moduleType;
  // uniqueName is used to generate unique names for the nodes in the BLIF file
  std::string uniqueName;
  // isBlackbox is used to determine if the module is a blackbox module
  bool isBlackbox = false;

  // Helper function to connect the input nodes of the current module
  // to the output nodes of the preceding module in the subject graph
  void connectInputNodesHelper(ChannelSignals &currentSignals,
                               BaseSubjectGraph *moduleBeforeSubjectGraph);

public:
  BaseSubjectGraph();
  // Constructor for a SubjectGraph based on an Operation
  BaseSubjectGraph(Operation *op);

  // Vectors and Maps to hold the input and output modules and SubjectGraphs.
  // All of this is necessary to connect the SubjectGraphs.
  // (input/output)Modules and ToResNum is populated the first time the
  // SubjectGraph is created.
  std::vector<Operation *> inputModules;
  std::vector<Operation *> outputModules;
  DenseMap<Operation *, unsigned int> inputModuleToResNum;
  DenseMap<Operation *, unsigned int> outputModuleToResNum;

  // These vectors and maps are populated after all of the SubjectGraphs are
  // created and while they are being connected. The position of a SubjectGraph
  // in the vector does not correspond to its Result Number. The position in the
  // vector is used to retrieve the Input/Output types such as data, address,
  // index etc. and Result Number is used to find which SubjectGraph to connect
  // to. An Operation might have multiple outputs, so it is
  // important to find the correct channel and input module.
  std::vector<BaseSubjectGraph *> inputSubjectGraphs;
  std::vector<BaseSubjectGraph *> outputSubjectGraphs;
  DenseMap<BaseSubjectGraph *, unsigned int> inputSubjectGraphToResNum;
  DenseMap<BaseSubjectGraph *, unsigned int> outputSubjectGraphToResNum;

  // A map from Operations to their corresponding BaseSubjectGraphs
  static inline DenseMap<Operation *, BaseSubjectGraph *> moduleMap;

  // A vector of all BaseSubjectGraphs. This is not a subset of the Values of
  // moduleMap, since not all of the SubjectGraphs are created from Operations.
  static inline std::vector<BaseSubjectGraph *> subjectGraphVector;

  // Holds the path to the directory of BLIF files
  static inline std::string baseBlifPath;

  // Holds the path to the BLIF file this subject graph is based on
  std::string fullPath;

  // Pointer to the LogicNetwork object that represents the BLIF file
  LogicNetwork *blifData;

  void appendVarsToPath(std::initializer_list<unsigned int> inputs);

  // Gets the MLIR Result Number between two SubjectGraphs
  static unsigned int getChannelNumber(BaseSubjectGraph *first,
                                       BaseSubjectGraph *second);

  // Populates inputSubjectGraphs and outputSubjectGraphs after all of the
  // SubjectGraphs are created
  void replaceOpsBySubjectGraph();

  virtual ~BaseSubjectGraph() = default;

  // Each Subject Graph implements its own connectInputNodes() function.
  // Retrieves the output nodes of its input module, and connects the nodes.
  virtual void connectInputNodes() = 0;
  // Returns the output nodes associated with a specific channel number (MLIR
  // Result Number)
  virtual ChannelSignals &returnOutputNodes(unsigned int resultNumber) = 0;
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

  // Inserts a pair of OEHB and TEHB buffers between two operations. Used to cut
  // loopbacks.
  static BufferPair createBuffers(Operation *inputOp, Operation *outputOp) {
    // Create OEHB buffer using the temporary string
    BufferSubjectGraph *oehb = new BufferSubjectGraph(
        inputOp, outputOp, getBufferTypeName(BufferType::OEHB));

    // Create TEHB buffer that connects to OEHB
    BufferSubjectGraph *tehb = new BufferSubjectGraph(
        oehb, outputOp, getBufferTypeName(BufferType::TEHB));

    return BufferPair(oehb, tehb);
  }

  void insertBuffer(BaseSubjectGraph *graph1, BaseSubjectGraph *graph2);
  void initBuffer();
};

// SubjectGraphGenerator class generates the Subject Graphs for the given
// FuncOp. Iterates through Ops and calls the appropriate SubjectGraph
// constructor.
class SubjectGraphGenerator {
public:
  SubjectGraphGenerator(handshake::FuncOp funcOp, StringRef blifFiles);
};

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_SUBJECT_GRAPH_H