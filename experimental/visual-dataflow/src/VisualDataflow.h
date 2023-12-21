//===- VisualDataflow.h - Godot-visible types -------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares types that must be visible by Godot in the GDExtension.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_VISUAL_DATAFLOW_VISUAL_DATAFLOW_H
#define DYNAMATIC_VISUAL_DATAFLOW_VISUAL_DATAFLOW_H

#include "Graph.h"
#include "GraphEdge.h"
#include "GraphNode.h"
#include "godot_cpp/classes/control.hpp"
#include "godot_cpp/classes/polygon2d.hpp"
#include <godot_cpp/classes/h_slider.hpp>
#include <godot_cpp/classes/label.hpp>
#include <godot_cpp/classes/line2d.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <vector>

using namespace dynamatic::experimental::visual_dataflow;

namespace godot {

/// Implements the logic to visualize the circuit in Godot
class VisualDataflow : public Control {
  GDCLASS(VisualDataflow, Control)

private:
  Graph graph;
  CycleNb cycle = -1;
  std::map<EdgeId, Line2D *> edgeIdToLine2D;
  std::map<EdgeId, std::vector<Line2D *>> edgeIdToLines;
  std::map<EdgeId, Polygon2D *> edgeIdToArrowHead;
  std::map<EdgeId, Label *> edgeIdToData;
  std::map<NodeId, PackedVector2Array> nodeIdToGodoPos;
  std::map<NodeId, Polygon2D *> nodeIdToPolygon;
  std::map<NodeId, std::vector<Line2D *>> nodeIdToContourLine;
  std::map<NodeId, bool> nodeIdToTransparency;
  std::map<EdgeId, size_t> edgeIdToTransparency;
  int nbClicked = 0;
  bool transparent = false;
  /// Initial colors corresponding to the different states on an edge
  std::vector<Color> stateColors = {
      Color(0.8, 0.0, 0.0, 1.0), Color(0.0, 0.0, 0.0, 1.0),
      Color(0.0, 0.0, 0.8, 1.0), Color(0.0, 0.8, 0.0, 1.0),
      Color(0.0, 0.8, 0.8, 1.0),
  };
  std::map<std::string, Color> colorNameToRGB = {
      {"lavender", Color(0.9, 0.9, 0.98, 1)},
      {"plum", Color(0.867, 0.627, 0.867, 1)},
      {"moccasin", Color(1.0, 0.894, 0.71, 1)},
      {"lightblue", Color(0.68, 0.85, 1.0, 1)},
      {"lightgreen", Color(0.56, 0.93, 0.56, 1)},
      {"coral", Color(1.0, 0.5, 0.31, 1)},
      {"gainsboro", Color(0.86, 0.86, 0.86, 1)},
      {"blue", Color(0, 0, 1, 1)},
      {"gold", Color(1.0, 0.843, 0.0, 1)},
      {"tan2", Color(1.0, 0.65, 0.0, 1)}};

  /// Godot object references
  Label *cycleLabel;
  HSlider *cycleSlider;

  /// Creates a graph from the circuits corresponding DOT and CSV file
  void createGraph(std::string inputDOTFile, std::string inputCSVFile);
  /// Draws each component of the graph in Godot
  void drawGraph();
  /// Changes the color of a given edge in function of its state
  void setEdgeColor(State state, std::vector<Line2D *> lines,
                    Polygon2D *arrowHead);

  void transparentEffect(double transparency);
  void highlightNode(NodeId nodeId);
  void transparentNode(NodeId nodeId);
  void drawBBs();
  void drawNodes();
  void drawEdges();

protected:
  /// Binds the cpp methods with Godot
  static void _bind_methods();

public:
  VisualDataflow();
  ~VisualDataflow() override = default;
  /// Given a circuits DOT and CSV file, draws the graph in its initial state in
  /// Godot
  void start(const godot::String &inputDOTFile,
             const godot::String &inputCSVFile);
  /// Draws in Godot the state of the graph in its next cycle
  void nextCycle();
  /// Draws in Godot the state of the graph in its previous cycle
  void previousCycle();
  /// Draws in Godot the state of the graph in a given cycle
  void changeCycle(int64_t cycleNb);
  /// Changes the corresponding color of a given edge state
  void changeStateColor(int64_t state, Color color);
  /// Detect if a node has been clicked and if so, highlight it
  void onClick(Vector2 position);
  /// Reset the selection of nodes
  void resetSelection();
}; // namespace godot

} // namespace godot

#endif // VISUAL_DATAFLOW_VISUAL_DATAFLOW_H
