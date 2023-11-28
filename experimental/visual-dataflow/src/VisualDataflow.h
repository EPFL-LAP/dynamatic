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
#include <godot_cpp/classes/h_slider.hpp>
#include <godot_cpp/classes/label.hpp>
#include <godot_cpp/classes/line2d.hpp>
#include <vector>

using namespace dynamatic::experimental::visual_dataflow;

namespace godot {

class VisualDataflow : public Control {
  GDCLASS(VisualDataflow, Control)

private:
  Graph graph;
  CycleNb cycle = 0;
  std::map<EdgeId, Line2D *> edgeIdToLine2D;

  // Godot object references
  Label *cycleLabel;
  HSlider *cycleSlider;

  void createGraph();
  void drawCycleNumber();
  void drawGraph();
  void setEdgeColor(State state, Line2D *line);

protected:
  static void _bind_methods();

public:
  VisualDataflow();

  ~VisualDataflow() override = default;

  void start();
  void nextCycle();
  void previousCycle();
  void changeCycle(int64_t cycleNb);
};

} // namespace godot

#endif // VISUAL_DATAFLOW_VISUAL_DATAFLOW_H
