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
#include "dynamatic/Support/DOT.h"
#include "godot_cpp/classes/control.hpp"
#include "godot_cpp/classes/h_slider.hpp"
#include "godot_cpp/classes/label.hpp"
#include "godot_cpp/classes/line2d.hpp"
#include "godot_cpp/classes/polygon2d.hpp"
#include "godot_cpp/classes/rich_text_label.hpp"
#include "godot_cpp/variant/color.hpp"
#include "godot_cpp/variant/string.hpp"
#include "godot_cpp/variant/vector2.hpp"
#include <vector>

namespace godot {

/// Implements the logic to visualize the circuit in Godot
class VisualDataflow : public Control {
  GDCLASS(VisualDataflow, Control)

public:
  /// Draws the graph representing a dataflow circuit from files that hold the
  /// circuit's information.
  /// - The DOT file gives the circuit's structure and node/edge positions.
  /// - The CSV file gives the state changes.
  void start(const godot::String &dotFilepath,
             const godot::String &csvFilepath);

  /// Draws in Godot the state of the graph in its next cycle
  void nextCycle();
  /// Draws in Godot the state of the graph in its previous cycle
  void previousCycle();
  /// Draws in Godot the state of the graph in a given cycle
  void gotoCycle(int64_t cycleNum);
  /// Changes the corresponding color of a given edge state
  void changeStateColor(int64_t state, Color color);
  /// Detect if a node has been clicked and if so, highlight it
  void onClick(Vector2 position);
  /// Reset the selection of nodes
  void resetSelection();

  ~VisualDataflow() override = default;

protected:
  /// Binds the cpp methods with Godot
  // NOLINTNEXTLINE(readability-identifier-naming)
  static void _bind_methods();

private:
  struct NodeGeometry {
    PackedVector2Array collision;
    Polygon2D *shape;
    std::vector<Line2D *> shapeLines;

    void setTransparency(double transparency);
  };

  struct EdgeGeometry {
    std::vector<Line2D *> segments;
    Polygon2D *arrowhead;
    RichTextLabel *data;

    void setTransparency(double transparency);

    void setColor(Color color);
  };

  /// The underlying graph we are visualizing.
  dynamatic::visual::GodotGraph graph;
  /// Visual elements to represent each node.
  mlir::DenseMap<const dynamatic::DOTGraph::Node *, NodeGeometry>
      nodeGeometries;
  /// Visual elements to represent each edge.
  mlir::DenseMap<const dynamatic::DOTGraph::Edge *, EdgeGeometry>
      edgeGeometries;

  /// Maximum number of cycles.
  unsigned maxCycle;
  /// Current cycle number, between 0 and `maxCycle` inclusive.
  unsigned cycle = 0;
  /// Set of currently selected nodes.
  mlir::DenseSet<const dynamatic::DOTGraph::Node *> selectedNodes;
  /// Maps currently selected edges to a boolean indicating whether the two
  /// nodes the edge connects are selected.
  mlir::DenseMap<const dynamatic::DOTGraph::Edge *, bool> selectedEdges;

  /// Colors for each possible channel state.
  std::unordered_map<dynamatic::visual::DataflowState, Color> stateColors = {
      {dynamatic::visual::DataflowState::UNDEFINED, Color(0.78, 0.0, 0.0, 1.0)},
      {dynamatic::visual::DataflowState::IDLE, Color(0.0, 0.0, 0.0, 1.0)},
      {dynamatic::visual::DataflowState::ACCEPT, Color(0.19, 0.14, 0.72, 1.0)},
      {dynamatic::visual::DataflowState::STALL, Color(0.85, 0.47, 0.14, 1.0)},
      {dynamatic::visual::DataflowState::TRANSFER, Color(0.0, 0.78, 0.0, 1.0)},
  };

  /// Godot object references
  Label *cycleLabel;
  HSlider *cycleSlider;

  /// Draws each component of the graph in Godot
  void drawGraph();
  /// Modifies the transparency of all graph elements
  void setGraphTransparency(double transparency);
  /// Draws the basic blocks of the graph
  void drawBBs();
  /// Draws the nodes of the graph
  void drawNodes();
  /// Draws the edges of the graph
  void drawEdges();
  /// Draw the current cycle.
  void drawCycle();
};

} // namespace godot

#endif // VISUAL_DATAFLOW_VISUAL_DATAFLOW_H
