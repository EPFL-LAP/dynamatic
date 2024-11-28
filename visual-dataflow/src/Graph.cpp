//===- Graph.cpp - Represents a graph ---------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of a Graph.
//
//===----------------------------------------------------------------------===//

#include "Graph.h"
#include "dynamatic/Support/DOT.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <set>
#include <sstream>
#include <utility>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::visual;

LogicalResult GodotGraph::fromDOTAndCSV(StringRef dotFilePath,
                                        StringRef csvFilePath) {
  return failure(failed(parseDOT(dotFilePath)) ||
                 failed(parseCSV(csvFilePath)));
}

template <typename T>
static LogicalResult defCallback(StringRef val, T &field) {
  field = val;
  return success();
}

template <bool Optional = false, typename T>
static LogicalResult
expectAttr(const DOTGraph::WithAttributes &withAttr, StringRef attrName,
           T &field, LogicalResult (*callback)(StringRef, T &) = &defCallback) {
  if (auto it = withAttr.attrs.find(attrName); it != withAttr.attrs.end())
    return callback(it->second, field);
  if constexpr (Optional) {
    return success();
  } else {
    llvm::errs() << "Expected DOT attribute '" << attrName << "'\n";
    return failure();
  }
}

template <typename T>
static LogicalResult toFloat(StringRef val, T &field) {
  field = std::stof(val.str());
  return success();
}

template <typename T>
static LogicalResult toInteger(StringRef val, T &field) {
  field = std::stoi(val.str());
  return success();
}

static LogicalResult toElemStyle(StringRef val, bool &style) {
  style = val == "dotted";
  return success();
}

static LogicalResult toSinglePos(StringRef val,
                                 std::pair<float, float> &position) {
  double x, y;
  char comma;
  std::istringstream iss(val.str());
  (iss >> x >> comma >> y);
  position = {x, y};
  return success();
}

static LogicalResult
toEdgePos(StringRef val, std::vector<std::pair<float, float>> &positions) {
  // All positions start with "e,", just discard this prefix
  std::istringstream iss(val.substr(2).str());
  std::string token;

  std::set<std::pair<float, float>> uniquePositions;
  while ((std::getline(iss, token, ' '))) {
    if (token.empty())
      continue;

    size_t commaPos = token.find(',');
    if (commaPos != std::string::npos) {
      std::string xStr = token.substr(0, commaPos);
      std::string yStr = token.substr(commaPos + 1);
      float x = std::stof(xStr);
      float y = std::stof(yStr);
      std::pair<float, float> position = std::make_pair(x, y);
      if (uniquePositions.insert(position).second)
        positions.push_back(position);
    }
  }
  return success();
}

static LogicalResult toBoundaries(StringRef val,
                                  std::vector<float> &boundaries) {
  std::stringstream ss(val.str());
  std::string item;
  while (std::getline(ss, item, ','))
    boundaries.push_back(std::stof(item));
  return success();
}

LogicalResult GodotGraph::parseDOT(StringRef filepath) {
  if (failed(graph.getBuilder().parseFromFile(filepath)))
    return failure();

  std::function<LogicalResult(const DOTGraph::Subgraph &, bool)>
      handleSubgraph =
          [&](const DOTGraph::Subgraph &sub, bool isRoot) -> LogicalResult {
    // Look at the attributes of each node to derive visual properties
    for (const DOTGraph::Node *node : sub.nodes) {
      NodeProps &props = nodes.try_emplace(node).first->second;
      if (failed(expectAttr(*node, "pos", props.position, toSinglePos)) ||
          failed(expectAttr(*node, "width", props.width, toFloat)) ||
          failed(expectAttr<true>(*node, "fillcolor", props.color)) ||
          failed(expectAttr<true>(*node, "shape", props.shape)) ||
          failed(expectAttr<true>(*node, "style", props.isDotted, toElemStyle)))
        return failure();
    }

    // Look at the attributes of each edge to derive visual properties
    for (const DOTGraph::Edge *edge : sub.edges) {
      EdgeProps &props = edges.try_emplace(edge).first->second;
      if (failed(expectAttr(*edge, "pos", props.positions, toEdgePos)) ||
          failed(expectAttr(*edge, "arrowhead", props.arrowhead)) ||
          failed(expectAttr(*edge, "from_idx", props.fromIdx, toInteger)) ||
          failed(expectAttr(*edge, "to_idx", props.toIdx, toInteger)) ||
          failed(expectAttr<true>(*edge, "style", props.isDotted, toElemStyle)))
        return failure();
    }

    if (!isRoot) { // Look at the attributes of the subgrap to derive visual
                   // properties
      SubgraphProps &props = subgraphs.try_emplace(&sub).first->second;
      if (failed(expectAttr(sub, "label", props.label)) ||
          failed(expectAttr(sub, "bb", props.boundaries, toBoundaries)) ||
          failed(expectAttr(sub, "lheight", props.labelSize.first, toFloat)) ||
          failed(expectAttr(sub, "lwidth", props.labelSize.second, toFloat)) ||
          failed(expectAttr(sub, "lp", props.labelPosition, toSinglePos)))
        return failure();
    }

    // Recurse on subgraphs
    for (const DOTGraph::Subgraph *subsub : sub.subgraphs) {
      if (failed(handleSubgraph(*subsub, false)))
        return failure();
    }
    return success();
  };

  return handleSubgraph(graph.getRoot(), true);
}

static const DenseMap<StringRef, DataflowState> STATE_DECODER = {
    {"undefined", DataflowState::UNDEFINED}, {"accept", DataflowState::ACCEPT},
    {"idle", DataflowState::IDLE},           {"stall", DataflowState::STALL},
    {"transfer", DataflowState::TRANSFER},
};

static LogicalResult decodeState(StringRef stateStr, DataflowState &state) {
  if (auto it = STATE_DECODER.find(stateStr); it != STATE_DECODER.end()) {
    state = it->second;
    return success();
  }
  return failure();
}

namespace {
struct CSVParser {

  using LineParser = std::function<ParseResult(CSVParser &)>;

  static ParseResult parse(StringRef filepath, const LineParser &callback,
                           bool columnNames = true) {
    CSVParser parser(filepath);

    std::ifstream file(filepath.str());
    if (!file.is_open()) {
      llvm::errs() << "Failed to open CSV file @ \"" << filepath << "\"\n";
      return failure();
    }

    std::string line;
    for (; std::getline(file, line); ++parser.lineNum, parser.colNum = 1) {
      // Ignore the first line which contain column names as well as empty lines
      if ((columnNames && parser.lineNum == 1) || line.empty())
        continue;
      parser.iss.str(line);
      parser.iss.clear();
      if (callback(parser))
        return failure();
    }
    return success();
  }

  ParseResult parseString(std::string &value) {
    value = StringRef{getNextColumn()}.trim().str();
    ++colNum;
    return success();
  };

  template <typename T>
  ParseResult parseInteger(T &value) {
    auto tok = getNextColumn();
    StringRef token = StringRef{tok}.trim();
    if (token.empty())
      return error("empty column, but expected positive integer");
    if (token.getAsInteger(10, value))
      return error("failed to parse token as positive integer");
    ++colNum;
    return success();
  };

  ParseResult error(StringLiteral msg) const {
    llvm::errs() << "Failed to parse CSV file @ \"" << filepath
                 << "\":" << lineNum << ":" << colNum << ": " << msg.str()
                 << "\n";
    return failure();
  }

private:
  std::string filepath;
  std::istringstream iss;
  unsigned lineNum = 1;
  unsigned colNum = 1;

  std::string getNextColumn() {
    std::string column;
    std::getline(iss, column, ',');
    return column;
  }

  CSVParser(StringRef filepath) : filepath(filepath) {}
};
} // namespace

LogicalResult GodotGraph::parseCSV(StringRef filepath) {
  transitions.emplace_back();
  CSVParser::LineParser callback = [&](CSVParser &parser) -> ParseResult {
    unsigned cycle, inPort, outPort;
    std::string srcNodeName, dstNodeName, stateString, data;
    if (parser.parseInteger(cycle) || parser.parseString(srcNodeName) ||
        parser.parseInteger(outPort) || parser.parseString(dstNodeName) ||
        parser.parseInteger(inPort) || parser.parseString(stateString) ||
        parser.parseString(data)) {
      return failure();
    }

    // Retrieve the state changes for the current cycle
    unsigned currentCycle = transitions.size() - 1;
    if (cycle != currentCycle) {
      // Copy the current state up until (and including) the new cycle
      for (unsigned i = currentCycle + 1; i <= cycle; ++i)
        transitions.push_back(transitions.back());
    }
    Transitions &states = transitions.back();

    // Decode the state
    DataflowState state;
    if (failed(decodeState(stateString, state)))
      return parser.error("failed to decode state");

    // Find the referenced nodes in the graph
    const DOTGraph::Node *srcNode = graph.getNode(srcNodeName);
    if (!srcNode) {
      llvm::errs() << "Node '" << srcNodeName << "' does not exist\n";
      return failure();
    }
    const DOTGraph::Node *dstNode = graph.getNode(dstNodeName);
    if (!dstNode) {
      llvm::errs() << "Node '" << dstNodeName << "' does not exist\n";
      return failure();
    }

    // Find the referenced edge in the graph
    for (const DOTGraph::Edge *edge : graph.getSuccessors(*srcNode)) {
      if (dstNode == edge->dstNode) {
        // Not only do the node names need to match, the port indices need to
        // match as well
        const EdgeProps &props = edges.at(edge);
        if (outPort == props.fromIdx && inPort == props.toIdx) {
          states[edge] = EdgeState{state, data};
          return success();
        }
      }
    }
    return parser.error("failed to find matching edge in graph");
  };
  auto ret = CSVParser::parse(filepath, callback);

  return ret;
}
