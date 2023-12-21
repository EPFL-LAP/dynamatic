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
#include <cstdint>
#include <godot_cpp/classes/area2d.hpp>
#include <godot_cpp/classes/canvas_item.hpp>
#include <godot_cpp/classes/control.hpp>
#include <godot_cpp/classes/node2d.hpp>
#include <godot_cpp/core/math.hpp>
#include <godot_cpp/core/memory.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace godot;
using namespace dynamatic;
using namespace dynamatic::experimental::visual_dataflow;

// Constant colors
const godot::Color TRANSPARENT_BLACK(0, 0, 0, 0.075);
const godot::Color OPAQUE_BLACK(0, 0, 0, 1.0);
const godot::Color OPAQUE_WHITE(1, 1, 1, 1.0);

void VisualDataflow::_bind_methods() {

  ClassDB::bind_method(D_METHOD("start", "inputDOTFile", "inputCSVFile"),
                       &VisualDataflow::start);
  ClassDB::bind_method(D_METHOD("nextCycle"), &VisualDataflow::nextCycle);
  ClassDB::bind_method(D_METHOD("previousCycle"),
                       &VisualDataflow::previousCycle);
  ClassDB::bind_method(D_METHOD("changeCycle", "cycleNb"),
                       &VisualDataflow::changeCycle);
  ClassDB::bind_method(D_METHOD("changeStateColor", "state", "color"),
                       &VisualDataflow::changeStateColor);
  ClassDB::bind_method(D_METHOD("onClick", "position"),
                       &VisualDataflow::onClick);
  ClassDB::bind_method(D_METHOD("resetSelection"),
                       &VisualDataflow::resetSelection);
}

VisualDataflow::VisualDataflow() = default;

void VisualDataflow::start(const godot::String &inputDOTFile,
                           const godot::String &inputCSVFile) {

  cycleLabel = (Label *)get_node_internal(
      NodePath("CanvasLayer/Timeline/MarginContainer/VBoxContainer/"
               "HBoxContainer/CycleNumber"));
  cycleSlider = (HSlider *)get_node_internal(
      NodePath("CanvasLayer/Timeline/MarginContainer/VBoxContainer/HSlider"));
  createGraph(inputDOTFile.utf8().get_data(), inputCSVFile.utf8().get_data());
  cycleSlider->set_max(graph.getCycleEdgeStates().size() - 1);
  drawGraph();
  changeCycle(0);
}

void VisualDataflow::createGraph(std::string inputDOTFile,
                                 std::string inputCSVFile) {

  GraphParser parser = GraphParser(&graph);

  if (failed(parser.parse(inputDOTFile))) {
    UtilityFunctions::printerr("Failed to parse the graph");
    return;
  }

  if (failed(parser.parse(inputCSVFile))) {
    UtilityFunctions::printerr("Failed to parse the graphs transitions");
    return;
  }
}

void VisualDataflow::drawBBs() {

  for (const auto &bb : graph.getBBs()) {

    std::vector<float> boundries = bb.boundries;
    Polygon2D *p = memnew(Polygon2D);
    PackedVector2Array points;

    points.push_back(Vector2(boundries.at(0), -boundries.at(1)));
    points.push_back(Vector2(boundries.at(2), -boundries.at(1)));
    points.push_back(Vector2(boundries.at(2), -boundries.at(3)));
    points.push_back(Vector2(boundries.at(0), -boundries.at(3)));

    p->set_polygon(points);
    p->set_color(TRANSPARENT_BLACK);

    add_child(p);

    Label *label = memnew(Label);
    label->set_text(bb.label.c_str());
    label->set_position(
        Vector2(bb.boundries.at(0) + 5,
                -bb.labelPosition.second - bb.labelSize.first * 35));
    label->add_theme_color_override("font_color", OPAQUE_BLACK);
    label->add_theme_font_size_override("font_size", 12);

    add_child(label);
  }
}

void VisualDataflow::drawNodes() {

  for (auto &node : graph.getNodes()) {

    std::pair<float, float> center = node.second.getPosition();
    float width = node.second.getWidth() * 70;
    float height = 35;

    Area2D *area2D = memnew(Area2D);
    Polygon2D *p = memnew(Polygon2D);
    PackedVector2Array points;
    Line2D *outline = memnew(Line2D);
    Vector2 firstPoint;

    if (node.second.getShape() == "diamond") {

      firstPoint = Vector2(center.first, -center.second + height / 2);
      Vector2 pt1 = Vector2(center.first, -center.second + height / 2);
      Vector2 pt2 = Vector2(center.first + width / 2, -center.second);
      Vector2 pt3 = Vector2(center.first, -center.second - height / 2);
      Vector2 pt4 = Vector2(center.first - width / 2, -center.second);

      points.push_back(pt1);
      points.push_back(pt2);
      points.push_back(pt3);
      points.push_back(pt4);

      p->set_polygon(points);
      PackedVector2Array nodePoints;

      nodePoints.append(pt1);
      nodePoints.append(pt2);
      nodePoints.append(pt3);
      nodePoints.append(pt4);

      nodeIdToGodoPos[node.first] = nodePoints;

    } else if (node.second.getShape() == "oval") {

      int numPoints = 30; // Adjust this for smoother oval

      for (int i = 0; i < numPoints; ++i) {
        float angle = 2 * M_PI * i / numPoints;
        float x = center.first + width / 2 * cos(angle);
        float y = -center.second + height / 2 * sin(angle);
        points.push_back(Vector2(x, y));

        if (i == 0)
          firstPoint = Vector2(x, y);
      }
      p->set_polygon(points);

      Vector2 pt1 = Vector2(center.first, -center.second + height / 2);
      Vector2 pt2 = Vector2(center.first + width / 2, -center.second);
      Vector2 pt3 = Vector2(center.first, -center.second - height / 2);
      Vector2 pt4 = Vector2(center.first - width / 2, -center.second);

      PackedVector2Array nodePoints;

      nodePoints.append(pt1);
      nodePoints.append(pt2);
      nodePoints.append(pt3);
      nodePoints.append(pt4);
      nodeIdToGodoPos[node.first] = nodePoints;

    } else {
      firstPoint =
          Vector2(center.first - width / 2, -center.second + height / 2);
      Vector2 pt1 =
          Vector2(center.first - width / 2, -center.second + height / 2);
      Vector2 pt2 =
          Vector2(center.first + width / 2, -center.second + height / 2);
      Vector2 pt3 =
          Vector2(center.first + width / 2, -center.second - height / 2);
      Vector2 pt4 =
          Vector2(center.first - width / 2, -center.second - height / 2);

      points.push_back(
          Vector2(center.first - width / 2, -center.second + height / 2));
      points.push_back(
          Vector2(center.first + width / 2, -center.second + height / 2));
      points.push_back(
          Vector2(center.first + width / 2, -center.second - height / 2));
      points.push_back(
          Vector2(center.first - width / 2, -center.second - height / 2));
      p->set_polygon(points);

      PackedVector2Array nodePoints;

      nodePoints.append(pt1);
      nodePoints.append(pt2);
      nodePoints.append(pt3);
      nodePoints.append(pt4);
      nodeIdToGodoPos[node.first] = nodePoints;
    }

    if (colorNameToRGB.count(node.second.getColor()))
      p->set_color(colorNameToRGB.at(node.second.getColor()));
    else
      p->set_color(OPAQUE_WHITE);

    Label *label = memnew(Label);
    label->set_text(node.second.getNodeId().c_str());
    label->add_theme_color_override("font_color", OPAQUE_BLACK);
    label->add_theme_font_size_override("font_size", 12);

    Vector2 size = label->get_combined_minimum_size();
    Vector2 newPosition =
        Vector2(center.first - size.x * 0.5, -(center.second + size.y * 0.5));
    label->set_position(newPosition);

    CenterContainer *centerContainer = memnew(CenterContainer);
    centerContainer->set_size(Vector2(width, height));
    centerContainer->set_position(
        Vector2(center.first - width / 2, -center.second - height / 2));
    centerContainer->add_child(label);
    area2D->add_child(p);
    area2D->add_child(centerContainer);

    nodeIdToPolygon[node.first] = p;

    std::vector<Line2D *> lines;

    if (node.second.getDashed()) {
      points.push_back(firstPoint);
      for (int i = 0; i < points.size() - 1; ++i) {

        Vector2 start = points[i];
        Vector2 end = points[i + 1];
        Vector2 segment = end - start;

        float segmentLength = segment.length();
        segment = segment.normalized();

        float currentLength = 0.0;
        while (currentLength < segmentLength) {
          Line2D *line = memnew(Line2D);
          line->set_width(1);
          line->set_default_color(OPAQUE_BLACK);
          Vector2 lineStart = start + segment * currentLength;
          Vector2 lineEnd =
              lineStart + segment * MIN(5, segmentLength - currentLength);
          PackedVector2Array pointsArray;

          pointsArray.append(lineStart);
          pointsArray.append(lineEnd);
          line->set_points(pointsArray);

          lines.push_back(line);

          area2D->add_child(line);

          currentLength += 10;
        }
      }
    } else {
      outline->set_points(points);
      outline->add_point(firstPoint);
      lines.push_back(outline);
    }

    outline->set_default_color(OPAQUE_BLACK);
    outline->set_width(1);
    area2D->add_child(outline);

    add_child(area2D);
    nodeIdToContourLine[node.first] = lines;
    nodeIdToTransparency[node.first] = false;
  }
}

void VisualDataflow::drawEdges() {

  for (auto &edge : graph.getEdges()) {

    Area2D *area2D = memnew(Area2D);
    std::vector<Line2D *> lines;
    Vector2 prev;
    Vector2 last;

    std::vector<std::pair<float, float>> positions = edge.getPositions();
    prev = Vector2(positions.at(1).first, -positions.at(1).second);
    last = prev;
    PackedVector2Array linePoints;

    for (size_t i = 1; i < positions.size(); ++i) {
      Vector2 point = Vector2(positions.at(i).first, -positions.at(i).second);
      linePoints.push_back(point);
      prev = last;
      last = point;
    }

    if (edge.getDashed()) {

      for (int i = 0; i < linePoints.size() - 1; ++i) {
        Vector2 start = linePoints[i];
        Vector2 end = linePoints[i + 1];
        Vector2 segment = end - start;
        float segmentLength = segment.length();
        segment = segment.normalized();

        float currentLength = 0.0;
        while (currentLength < segmentLength) {
          Line2D *line = memnew(Line2D);
          line->set_width(1);
          line->set_default_color(OPAQUE_WHITE);
          Vector2 lineStart = start + segment * currentLength;
          Vector2 lineEnd =
              lineStart + segment * MIN(5, segmentLength - currentLength);

          PackedVector2Array pointsArray;
          pointsArray.append(lineStart);
          pointsArray.append(lineEnd);
          line->set_points(pointsArray);

          area2D->add_child(line);
          lines.push_back(line);

          currentLength += 5 + 5;
        }
      }

    } else {
      Line2D *line = memnew(Line2D);
      line->set_points(linePoints);
      line->set_default_color(OPAQUE_WHITE);
      line->set_width(1);

      area2D->add_child(line);
      lines.push_back(line);
    }

    edgeIdToLines[edge.getEdgeId()] = lines;

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
    arrowHead->set_color(OPAQUE_BLACK);
    area2D->add_child(arrowHead);
    edgeIdToArrowHead[edge.getEdgeId()] = arrowHead;

    Label *label = memnew(Label);
    label->set_text("");
    label->set_position(prev);
    label->add_theme_color_override("font_color", OPAQUE_BLACK);
    label->add_theme_font_size_override("font_size", 10);
    add_child(label);
    edgeIdToData[edge.getEdgeId()] = label;

    add_child(area2D);
    edgeIdToTransparency[edge.getEdgeId()] = 0;
  }
}

void VisualDataflow::drawGraph() {

  drawBBs();
  drawNodes();
  drawEdges();
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
      std::map<EdgeId, std::pair<State, Data>> edgeStates =
          graph.getCycleEdgeStates().at(cycle);
      for (auto &edgeState : edgeStates) {

        EdgeId edgeId = edgeState.first;
        State state = edgeState.second.first;
        std::vector<Line2D *> lines = edgeIdToLines[edgeId];
        Polygon2D *arrowHead = edgeIdToArrowHead[edgeId];

        setEdgeColor(state, lines, arrowHead);
        edgeIdToData.at(edgeId)->set_text(edgeState.second.second.c_str());
      }
    }
  }
}

void VisualDataflow::setEdgeColor(State state, std::vector<Line2D *> lines,
                                  Polygon2D *arrowHead) {
  Color color = stateColors.at(state);

  for (auto &line : lines) {
    color.a = line->get_default_color().a;
    line->set_default_color(color);
  }

  arrowHead->set_color(color);
}

void VisualDataflow::changeStateColor(int64_t state, Color color) {
  State stateEnum;
  if (state == 0) {
    stateEnum = State::UNDEFINED;
  } else if (state == 1) {
    stateEnum = State::ACCEPT;
  } else if (state == 2) {
    stateEnum = State::IDLE;
  } else if (state == 3) {
    stateEnum = State::STALL;
  } else if (state == 4) {
    stateEnum = State::TRANSFER;
  } else {
    UtilityFunctions::printerr("Invalid state");
    return;
  }

  stateColors.at(state) = color;

  auto cycleStates = graph.getCycleEdgeStates();
  for (auto &edgeState : cycleStates.at(cycle)) {
    EdgeId edgeId = edgeState.first;
    State edgeStateEnum = edgeState.second.first;
    if (edgeStateEnum == stateEnum) {
      std::vector<Line2D *> lines = edgeIdToLines[edgeId];
      Polygon2D *arrowHead = edgeIdToArrowHead[edgeId];
      setEdgeColor(stateEnum, lines, arrowHead);
    }
  }
}

double crossProduct(Vector2 a, Vector2 b, Vector2 c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

bool isInside(Vector2 p, Vector2 a, Vector2 b, Vector2 c, Vector2 d) {
  return crossProduct(a, b, p) * crossProduct(c, d, p) >= 0 &&
         crossProduct(b, c, p) * crossProduct(d, a, p) >= 0;
}

void VisualDataflow::onClick(Vector2 position) {
  for (auto &elem : nodeIdToGodoPos) {

    PackedVector2Array points = elem.second;

    if (isInside(position, points[0], points[1], points[2], points[3])) {
      if (nbClicked == 0) {
        ++nbClicked;
        transparentEffect(0.3);
        highlightNode(elem.first);
      } else {
        if (nodeIdToTransparency[elem.first]) {
          ++nbClicked;
          highlightNode(elem.first);
        } else {
          --nbClicked;

          if (nbClicked == 0) {
            transparentEffect(1);
          } else {
            transparentNode(elem.first);
          }
        }
      }
    }
  }
}

void VisualDataflow::transparentEffect(double transparency) {

  for (auto &elem : nodeIdToPolygon) {
    Color color = elem.second->get_color();
    color.a = transparency;
    elem.second->set_color(color);
    nodeIdToTransparency[elem.first] = (transparency < 1);
  }

  for (auto &elem : nodeIdToContourLine) {
    for (auto &line : elem.second) {
      Color color = line->get_default_color();
      color.a = transparency;
      line->set_default_color(color);
    }
  }

  for (auto &elem : edgeIdToArrowHead) {
    Color color = elem.second->get_color();
    color.a = transparency;
    elem.second->set_color(color);
    edgeIdToTransparency[elem.first] = 0;
  }

  for (auto &elem : edgeIdToLines) {
    for (auto &line : elem.second) {
      Color color = line->get_default_color();
      color.a = transparency;
      line->set_default_color(color);
    }
  }
}

void VisualDataflow::highlightNode(const NodeId &nodeId) {

  Color c = nodeIdToPolygon[nodeId]->get_color();
  c.a = 1;
  nodeIdToPolygon[nodeId]->set_color(c);
  for (auto &line : nodeIdToContourLine[nodeId]) {
    Color c3 = line->get_default_color();
    c3.a = 1;
    line->set_default_color(c3);
  }
  nodeIdToTransparency[nodeId] = false;
  for (auto &edge : graph.getInOutEdgesOfNode(nodeId)) {
    ++edgeIdToTransparency[edge];
    Color c2 = edgeIdToArrowHead[edge]->get_color();
    c2.a = 1;
    edgeIdToArrowHead[edge]->set_color(c2);
    for (auto &line : edgeIdToLines[edge]) {
      Color c3 = line->get_default_color();
      c3.a = 1;
      line->set_default_color(c3);
    }
  }
}

void VisualDataflow::transparentNode(const NodeId &nodeId) {

  Color c = nodeIdToPolygon[nodeId]->get_color();
  c.a = 0.3;
  nodeIdToPolygon[nodeId]->set_color(c);
  for (auto &line : nodeIdToContourLine[nodeId]) {
    Color c3 = line->get_default_color();
    c3.a = 0.3;
    line->set_default_color(c3);
  }
  nodeIdToTransparency[nodeId] = true;
  for (auto &edge : graph.getInOutEdgesOfNode(nodeId)) {
    --edgeIdToTransparency[edge];
    if (edgeIdToTransparency[edge] == 0) {
      Color c2 = edgeIdToArrowHead[edge]->get_color();
      c2.a = 0.3;
      edgeIdToArrowHead[edge]->set_color(c2);
      for (auto &line : edgeIdToLines[edge]) {
        Color c3 = line->get_default_color();
        c3.a = 0.3;
        line->set_default_color(c3);
      }
    }
  }
}

void VisualDataflow::resetSelection() {
  nbClicked = 0;
  transparentEffect(1);
}
