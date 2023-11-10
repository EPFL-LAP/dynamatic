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
#include "godot_cpp/classes/center_container.hpp"
#include "godot_cpp/classes/control.hpp"
#include "godot_cpp/classes/global_constants.hpp"
#include "godot_cpp/classes/label.hpp"
#include "godot_cpp/classes/line2d.hpp"
#include "godot_cpp/classes/node.hpp"
#include "godot_cpp/classes/panel.hpp"
#include "godot_cpp/classes/polygon2d.hpp"
#include "godot_cpp/core/class_db.hpp"
#include "godot_cpp/variant/packed_vector2_array.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include <godot_cpp/classes/control.hpp>
#include <godot_cpp/core/memory.hpp>
#include <godot_cpp/variant/vector2.hpp>

using namespace llvm;
using namespace mlir;
using namespace godot;
using namespace dynamatic;
using namespace dynamatic::experimental::visual_dataflow;

void VisualDataflow::_bind_methods() {
  ClassDB::bind_method(D_METHOD("addPanel"), &VisualDataflow::addPanel);
}

VisualDataflow::VisualDataflow() = default;

void VisualDataflow::my_process(double delta) {}

void VisualDataflow::addPanel() {
  Graph graph = Graph();
  GraphParser parser = GraphParser(
      "/home/alicepotter/dynamatic/experimental/visual-dataflow/test/bicg.dot");
  if (failed(parser.parse(&graph))) {
    return;
  }

  size_t nodeCounter = 0;

  for (auto &node : graph.getNodes()) {
    nodeCounter++;
    Panel *panel = memnew(Panel);
    panel->set_custom_minimum_size(Vector2(10, 10));
    std::pair<float, float> pos = node.second.getPosition();
    panel->set_position(Vector2(pos.first, 2554 - pos.second));

    // Create a center container to hold the label
    CenterContainer *center_container = memnew(CenterContainer);
    center_container->set_anchor(SIDE_LEFT, ANCHOR_BEGIN);
    center_container->set_anchor(SIDE_TOP, ANCHOR_BEGIN);
    center_container->set_anchor(SIDE_RIGHT, ANCHOR_END);
    center_container->set_anchor(SIDE_BOTTOM, ANCHOR_END);
    panel->add_child(center_container);

    // Add the label to the center container
    Label *node_label = memnew(Label);
    node_label->set_text(node.second.getNodeId().c_str());
    node_label->set_horizontal_alignment(
        HorizontalAlignment::HORIZONTAL_ALIGNMENT_CENTER);
    node_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD);
    center_container->add_child(node_label);

    add_child(panel);
  }

  for (auto &edge : graph.getEdges()) {
    Line2D *line = memnew(Line2D);
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
    points.push_back(last);
    if (prev.x == last.x) {
      if (prev.y < last.y) {
        points.push_back(Vector2(last.x - 10, last.y - 10));
        points.push_back(Vector2(last.x + 10, last.y - 10));
      } else {
        points.push_back(Vector2(last.x - 10, last.y + 10));
        points.push_back(Vector2(last.x + 10, last.y + 10));
      }

    } else {
      if (prev.x < last.x) {
        points.push_back(Vector2(last.x - 10, last.y + 10));
        points.push_back(Vector2(last.x - 10, last.y - 10));
      } else {
        points.push_back(Vector2(last.x + 10, last.y + 10));
        points.push_back(Vector2(last.x + 10, last.y - 10));
      }
    }
    arrowHead->set_polygon(points);
    line->add_child(arrowHead);
    line->set_width(3);
    add_child(line);
  }

  Line2D *line = memnew(Line2D);
  line->set_width(3);
  Vector2 start = Vector2(0, 0);
  Vector2 end = Vector2(0, 100);
  line->add_point(start);
  line->add_point(end);
  Polygon2D *arrowHead = memnew(Polygon2D);
  PackedVector2Array points;
  points.push_back(end);
  points.push_back(Vector2(-15, 80));
  points.push_back(Vector2(15, 80));
  arrowHead->set_polygon(points);
  line->add_child(arrowHead);
  add_child(line);

  // arrow_head.set_polygon(const PackedVector2Array &polygon)
}
