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

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
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

struct NodeProcessingRule {
  std::string pattern;
  ChannelSignals &signals;
  bool renameNode;
  std::function<void(Node *)> extraProcessing;
};

/// Base class for all subject graphs. This class represents the subject graph
/// of an Operation in MLIR.
/// Each BaseSubjectGraph maintains information about:
/// - Input/output connections to other modules and subject graphs
/// - Signals for handshake protocols (data, valid, ready signals)
/// - Unique naming and module type identification
class BaseSubjectGraph {
protected:
  // Operation that the SubjectGraph is based on
  Operation *op;

  // uniqueName is used to generate unique names for the nodes in the BLIF file
  std::string uniqueName;

  // isBlackbox is used to determine if the module is a blackbox module
  bool isBlackbox = false;

  void loadBlifFile(std::initializer_list<unsigned int> inputs,
                    std::string toAppend = "");

  // Helper function to connect the input nodes of the current module
  // to the output nodes of the preceding module in the subject graph
  void connectInputNodesHelper(ChannelSignals &currentSignals,
                               BaseSubjectGraph *moduleBeforeSubjectGraph);

  // Function to assign signals to the ChannelSignals struct based on the
  // rules provided. It processes the nodes in the BLIF file and assigns them
  // to the appropriate ChannelSignals based on the pattern specified in the
  // NodeProcessingRule.
  void processNodesWithRules(const std::vector<NodeProcessingRule> &rules);

  // Inserts a subject graph in between two other subject graphs.
  void insertNewSubjectGraph(BaseSubjectGraph *predecessorGraph,
                             BaseSubjectGraph *successorGraph);

public:
  // Default constructor, used for Subject Graph creation without an MLIR
  // Operation.
  BaseSubjectGraph();
  // Constructor for a SubjectGraph based on an Operation
  BaseSubjectGraph(Operation *op);

  // The populated maps store channel-specific information that connects
  // SubjectGraphs together. Each SubjectGraph has a position in the vector
  // which determines what type of data it handles (e.g., data, address, index).
  // Separately, a Result Number identifies which specific output channel of an
  // Operation to connect to, since an Operation may have multiple inputs and
  // outpus. The maps are essential for establishing the correct connections
  // between SubjectGraphs, while the vectors determine the data types being
  // passed through those connections.
  std::vector<BaseSubjectGraph *> inputSubjectGraphs;
  std::vector<BaseSubjectGraph *> outputSubjectGraphs;
  DenseMap<BaseSubjectGraph *, unsigned int> inputSubjectGraphToResultNumber;

  // static map that holds all Operation/Subject Graph Pairs
  static inline DenseMap<Operation *, BaseSubjectGraph *> moduleMap;

  // A vector of all BaseSubjectGraphs. This is not a subset of the Values of
  // moduleMap, since not all of the SubjectGraphs are created from Operations
  // (Buffers will be inserted to ensure acyclicity).
  static inline std::vector<BaseSubjectGraph *> subjectGraphVector;

  // Pointer to the LogicNetwork object that represents the BLIF file
  LogicNetwork *blifData;

  // Populates inputSubjectGraphs and outputSubjectGraphs after all of the
  // SubjectGraphs are created
  void buildSubjectGraphConnections();

  virtual ~BaseSubjectGraph() = default;

  // Each Subject Graph implements its own connectInputNodes() function.
  // Retrieves the output nodes of its input module, and connects the nodes.
  virtual void connectInputNodes() = 0;

  // Returns the output nodes associated with a specific MLIR result number.
  virtual ChannelSignals &returnOutputNodes(unsigned int channelIndex) = 0;
};

/// Below is class definitions of Subject Graph for supported modules. Each
/// class has its own definitions for connectInputNodes() and
/// returnOutputNodes() functions. The variables and the functions of the
/// classes are specialized to its hardware description.

class ArithSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int dataWidth = 0;
  ChannelSignals lhsNodes;
  ChannelSignals rhsNodes;
  ChannelSignals resultNodes;

public:
  ArithSubjectGraph(Operation *op);
  void connectInputNodes() override;
  ChannelSignals &returnOutputNodes(unsigned int channelIndex) override;
};

class ForkSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int size = 0;
  unsigned int dataWidth = 0;
  ChannelSignals inputNodes;
  std::vector<ChannelSignals> outputNodes;

  void processOutOfRuleNodes();

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
  std::vector<ChannelSignals> inputNodes;
  ChannelSignals indexNodes;
  ChannelSignals outputNodes;

  void processOutOfRuleNodes();

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
  std::vector<ChannelSignals> inputNodes;
  ChannelSignals indexNodes;
  ChannelSignals outputNodes;

  void processOutOfRuleNodes();

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
  ChannelSignals trueOut;
  ChannelSignals falseOut;

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
  ChannelSignals condition;
  ChannelSignals trueValue;
  ChannelSignals falseValue;
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
  std::vector<ChannelSignals> inputNodes;
  ChannelSignals outputNodes;

  void processOutOfRuleNodes();

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

class BufferSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int dataWidth = 0;
  ChannelSignals inputNodes;
  ChannelSignals outputNodes;
  std::string bufferType;

public:
  BufferSubjectGraph(Operation *op);
  BufferSubjectGraph(unsigned int inputDataWidth, std::string bufferTypeName);

  void connectInputNodes() override;
  ChannelSignals &returnOutputNodes(unsigned int) override;

  // Creates a new Buffer SubjectGraph, and inserts it into the middle of 2
  // other SubjectGraphs, given by the input and output Ops.
  static void createAndInsertNewBuffer(Operation *inputOp, Operation *outputOp,
                                       std::string bufferTypeName) {
    // Get the SubjectGraphs of the operations
    BaseSubjectGraph *graph1 = moduleMap[inputOp];
    BaseSubjectGraph *graph2 = moduleMap[outputOp];

    // Get the result number between inputOp and outputOp, so that we place the
    // buffer on the correct channel
    unsigned int resultNum = graph2->inputSubjectGraphToResultNumber[graph1];
    ChannelSignals &channel = graph1->returnOutputNodes(resultNum);

    // Data width of the channel between the Ops. New buffer should have the
    // same data width.
    unsigned int inputDataWidth = channel.dataSignals.size();

    // Create the new buffer.
    BufferSubjectGraph *breakDvr =
        new BufferSubjectGraph(inputDataWidth, std::move(bufferTypeName));

    // Insert the new BufferSubjectGraph in between the 2 SubjectGraphs
    breakDvr->insertNewSubjectGraph(graph1, graph2);
  }

  // Initializes the BufferSubjectGraph by loading the BLIF file based on the
  // buffer type name and data width.
  void initBuffer();
};

// SubjectGraphGenerator function generates the Subject Graphs for the given
// FuncOp. Iterates through Ops and calls the appropriate SubjectGraph
// constructor.
void subjectGraphGenerator(handshake::FuncOp funcOp, StringRef blifFiles);

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_SUBJECT_GRAPH_H
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
