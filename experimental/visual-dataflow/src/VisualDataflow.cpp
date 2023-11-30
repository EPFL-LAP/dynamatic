//===- VisualDataflow.cpp - Godot-visible types -----------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines Godot-visible data types.
//
//===----------------------------------------------------------------------===//

#include "VisualDataflow.h"
#include "Graph.h"
#include "GraphParser.h"
#include "dynamatic/Support/DOTPrinter.h"
#include "dynamatic/Support/TimingModels.h"
#include "godot_cpp/classes/canvas_layer.hpp"
#include "godot_cpp/classes/center_container.hpp"
#include "godot_cpp/classes/color_rect.hpp"
#include "godot_cpp/classes/control.hpp"
#include "godot_cpp/classes/font.hpp"
#include "godot_cpp/classes/global_constants.hpp"
#include "godot_cpp/classes/label.hpp"
#include "godot_cpp/classes/line2d.hpp"
#include "godot_cpp/classes/node.hpp"
#include "godot_cpp/classes/panel.hpp"
#include "godot_cpp/classes/polygon2d.hpp"
#include "godot_cpp/classes/style_box_flat.hpp"
#include "godot_cpp/core/class_db.hpp"
#include "godot_cpp/variant/packed_vector2_array.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include <godot_cpp/classes/canvas_item.hpp>
#include <godot_cpp/classes/control.hpp>
#include <godot_cpp/core/math.hpp>
#include <godot_cpp/core/memory.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/variant/vector2.hpp>

using namespace llvm;
using namespace mlir;
using namespace godot;
using namespace dynamatic;
using namespace dynamatic::experimental::visual_dataflow;

void VisualDataflow::_bind_methods() {
  ClassDB::bind_method(D_METHOD("start"), &VisualDataflow::start);
  ClassDB::bind_method(D_METHOD("nextCycle"), &VisualDataflow::nextCycle);
  ClassDB::bind_method(D_METHOD("previousCycle"),
                       &VisualDataflow::previousCycle);
  ClassDB::bind_method(D_METHOD("changeCycle", "cycleNb"),
                       &VisualDataflow::changeCycle);
}

VisualDataflow::VisualDataflow() = default;

void VisualDataflow::start() {
  cycleLabel = (Label *)get_node_internal(NodePath(
      "CanvasLayer/MarginContainer/VBoxContainer/HBoxContainer/CycleNumber"));
  cycleSlider = (HSlider *)get_node_internal(
      NodePath("CanvasLayer/MarginContainer/VBoxContainer/HSlider"));
  createGraph();
  cycleSlider->set_max(graph.getCycleEdgeStates().size() - 1);
  drawGraph();
  changeCycle(0);
}

void VisualDataflow::createGraph() {
  GraphParser parser = GraphParser(&graph);

  std::string inputDOTFile = "../test/bicg.dot";
  std::string inputCSVFile = "../test/bicg.csv";

  if (failed(parser.parse(inputDOTFile))) {
    UtilityFunctions::printerr("Failed to parse graph");
    return;
  }

  if (failed(parser.parse(inputCSVFile))) {
    UtilityFunctions::printerr("Failed to parse transitions");
    return;
  }
}

void VisualDataflow::drawGraph() {
  std::map<std::string, Color> mapColor;
  mapColor["lavender"] = Color(0.9, 0.9, 0.98, 1);
  mapColor["plum"] = Color(0.867, 0.627, 0.867, 1);
  mapColor["moccasin"] = Color(1.0, 0.894, 0.71, 1);
  mapColor["lightblue"] = Color(0.68, 0.85, 1.0, 1);
  mapColor["lightgreen"] = Color(0.56, 0.93, 0.56, 1);
  mapColor["coral"] = Color(1.0, 0.5, 0.31, 1);
  mapColor["gainsboro"] = Color(0.86, 0.86, 0.86, 1);
  mapColor["blue"] = Color(0, 0, 1, 1);
  mapColor["gold"] = Color(1.0, 0.843, 0.0, 1);
  mapColor["tan2"] = Color(1.0, 0.65, 0.0, 1);

  for (auto &node : graph.getNodes()) {
    Panel *panel = memnew(Panel);
    StyleBoxFlat *style = memnew(StyleBoxFlat);
    if (mapColor.count(node.second.getColor()))
      style->set_bg_color(mapColor.at(node.second.getColor()));
    else
      style->set_bg_color(Color(1, 1, 1, 1));
    panel->add_theme_stylebox_override("panel", style);
    panel->set_custom_minimum_size(
        Vector2(node.second.getWidth() * 70, 0.5 * 70));
    std::pair<float, float> pos = node.second.getPosition();
    panel->set_position(Vector2(pos.first - node.second.getWidth() * 35,
                                2554 - pos.second - 0.5 * 35));

    // Create a center container to hold the label
    CenterContainer *center_container = memnew(CenterContainer);
    center_container->set_anchor(SIDE_LEFT, ANCHOR_BEGIN);
    center_container->set_anchor(SIDE_TOP, ANCHOR_BEGIN);
    center_container->set_anchor(SIDE_RIGHT, ANCHOR_END);
    center_container->set_anchor(SIDE_BOTTOM, ANCHOR_END);
    panel->add_child(center_container);

    /// Add the label to the center container
    Label *node_label = memnew(Label);
    node_label->set_text(node.second.getNodeId().c_str());
    node_label->add_theme_color_override(
        "font_color", Color(0, 0, 0)); // Change to font_color
    node_label->set_horizontal_alignment(
        HorizontalAlignment::HORIZONTAL_ALIGNMENT_CENTER);
    node_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD);
    node_label->add_theme_font_size_override("font_size", 13);

    // // Create a new Font instance and set its properties
    //     Ref<Font> custom_font = memnew(Font);
    //     custom_font->set("size", 20); // Set your desired size here

    // Apply the custom font to the label
    // node_label->add_theme_font_override("font", custom_font);

    center_container->add_child(node_label);

    add_child(panel);
  }

  for (auto &edge : graph.getEdges()) {
    Line2D *line = memnew(Line2D);
    line->set_default_color(Color(0, 0, 0, 1));
    std::vector<std::pair<float, float>> positions = edge.getPositions();
    Vector2 prev =
        Vector2(positions.at(1).first, 2554 - positions.at(1).second);
    Vector2 last = prev;
    for (size_t i = 1; i < positions.size(); ++i) {
      Vector2 point =
          Vector2(positions.at(i).first, 2554 - positions.at(i).second);
      line->add_point(point);
      prev = last;
      last = point;
    }
    Polygon2D *arrowHead = memnew(Polygon2D);
    PackedVector2Array points;
    if (prev.x == last.x) {
      points.push_back(Vector2(last.x - 8, last.y));
      points.push_back(Vector2(last.x + 8, last.y));
      if (prev.y < last.y) {
        // arrow pointing to the bottom
        points.push_back(Vector2(last.x, last.y + 12));
      } else {
        // arrow pointing to the top
        points.push_back(Vector2(last.x, last.y - 12));
      }

    } else {
      points.push_back(Vector2(last.x, last.y + 8));
      points.push_back(Vector2(last.x, last.y - 8));
      if (prev.x < last.x) {
        // arrow poiting to the right
        points.push_back(Vector2(last.x + 12, last.y));
      } else {
        // arrow pointing to the left
        points.push_back(Vector2(last.x - 12, last.y));
      }
    }
    arrowHead->set_polygon(points);
    arrowHead->set_color(Color(0, 0, 0, 1));
    line->add_child(arrowHead);
    line->set_width(1.5);
    add_child(line);
    edgeIdToLine2D[edge.getEdgeId()] = line;
  }

  for (const auto &bb : graph.getBBs()){
    Line2D *line = memnew(Line2D);
    line->set_default_color(Color(1, 0, 0, 1));
    std::vector<float> boundries = bb.boundries;
    line->add_point(Vector2(boundries.at(0), boundries.at(1)));
    line->add_point(Vector2(boundries.at(2), boundries.at(1)));
    line->add_point(Vector2(boundries.at(2), boundries.at(3)));
    line->add_point(Vector2(boundries.at(0), boundries.at(3)));
    line->add_point(Vector2(boundries.at(0), boundries.at(1)));

    line->set_width(2.5);

    add_child(line);


  }
}

void VisualDataflow::nextCycle() {
  if (cycle < cycleSlider->get_max()) {
    changeCycle(cycle + 1);
  }
}

void VisualDataflow::previousCycle() {
  if (cycle > 0) {
    changeCycle(cycle - 1);
  }
}

void VisualDataflow::changeCycle(int64_t cycleNb) {
  if (cycle != cycleNb) {
    cycle = Math::min(Math::max((double)cycleNb, 0.0), cycleSlider->get_max());
    cycleLabel->set_text("Cycle: " + String::num_int64(cycle));
    cycleSlider->set_value(cycle);

    if (graph.getCycleEdgeStates().count(cycle)) {
      std::map<EdgeId, State> edgeStates = graph.getCycleEdgeStates().at(cycle);
      for (auto &edgeState : edgeStates) {
        EdgeId edgeId = edgeState.first;
        State state = edgeState.second;
        Line2D *line = edgeIdToLine2D[edgeId];
        setEdgeColor(state, line);
      }
    }
  }
}

void VisualDataflow::setEdgeColor(State state, Line2D *line) {
  Color color = Color(0, 0, 0, 1);
  if (state == UNDEFINED) {
    color = Color(0.8, 0, 0, 1);
  } else if (state == READY) {
    color = Color(0, 0, 0.8, 1);
  } else if (state == EMPTY) {
    color = Color(0, 0, 0, 1);
  } else if (state == VALID) {
    color = Color(0, 0.8, 0, 1);
  } else if (state == VALID_READY) {
    color = Color(0, 0.8, 0.8, 1);
  }
  line->set_default_color(color);
  for (int i = 0; i < line->get_child_count(); ++i) {
    Node *child = line->get_child(i);
    if (child->get_class() == "Polygon2D") {
      Polygon2D *arrowHead = (Polygon2D *)child;
      arrowHead->set_color(color);
    }
  }
}
