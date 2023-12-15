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

class VisualDataflow : public Control {
  GDCLASS(VisualDataflow, Control)

private:
  Graph graph;
  CycleNb cycle = -1;
  std::map<EdgeId, Line2D *> edgeIdToLine2D;

  std::map<EdgeId, std::vector<Line2D *>> edgeIdToLines;
  std::map<EdgeId, Polygon2D *> edgeIdToArrowHead;
  std::map<EdgeId, Label *> edgeIdToData;
  std::vector<Color> stateColors = {
      Color(0.8, 0.0, 0.0, 1.0), Color(0.0, 0.0, 0.0, 1.0),
      Color(0.0, 0.0, 0.8, 1.0), Color(0.0, 0.8, 0.0, 1.0),
      Color(0.0, 0.8, 0.8, 1.0),
  };

  // Godot object references
  Label *cycleLabel;
  HSlider *cycleSlider;

  void createGraph(std::string inputDOTFile, std::string inputCSVFile);
  void drawCycleNumber();
  void drawGraph();
  void setEdgeColor(State state, std::vector<Line2D *> lines,
                    Polygon2D *arrowHead);

protected:
  static void _bind_methods();

public:
  VisualDataflow();

  ~VisualDataflow() override = default;

  void start(godot::String inputDOTFile, godot::String inputCSVFile);
  void nextCycle();
  void previousCycle();
  void changeCycle(int64_t cycleNb);
  void changeStateColor(int64_t state, Color color);
};

} // namespace godot

#endif // VISUAL_DATAFLOW_VISUAL_DATAFLOW_H
