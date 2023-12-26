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

const godot::Color TRANSPARENT_BLACK(0, 0, 0, 0.075);
const godot::Color OPAQUE_BLACK(0, 0, 0, 1.0);
const godot::Color OPAQUE_WHITE(1, 1, 1, 1.0);

const int NODE_HEIGHT = 35;
const int NODE_WIDTH_SCALING_COEFFICIENT = 70;
const int DASH_LENGTH = 5;

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

void createDashedLine(PackedVector2Array &points, std::vector<Line2D *> *lines,
                      Area2D *area2D) {
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
          lineStart + segment * MIN(DASH_LENGTH, segmentLength - currentLength);
      PackedVector2Array linePoints;
      linePoints.append(lineStart);
      linePoints.append(lineEnd);
      line->set_points(linePoints);
      lines->push_back(line);
      area2D->add_child(line);
      currentLength += 2 * DASH_LENGTH;
    }
  }
}

void VisualDataflow::drawNodes() {

  for (auto &node : graph.getNodes()) {
    std::pair<float, float> center = node.second.getPosition();
    float width = node.second.getWidth() * NODE_WIDTH_SCALING_COEFFICIENT;
    Area2D *area2D = memnew(Area2D);
    Polygon2D *godotNode = memnew(Polygon2D);
    PackedVector2Array points;
    Line2D *outline = memnew(Line2D);
    Vector2 firstPoint, point1, point2, point3, point4;
    Shape shape = node.second.getShape();

    // Set points for either diamond or box shapes
    if (shape == "diamond" || shape == "box") {
      if (shape == "box") {
        // Define points for a box-shaped node
        point1 =
            Vector2(center.first - width / 2, -center.second + NODE_HEIGHT / 2);
        point2 =
            Vector2(center.first + width / 2, -center.second + NODE_HEIGHT / 2);
        point3 =
            Vector2(center.first + width / 2, -center.second - NODE_HEIGHT / 2);
        point4 =
            Vector2(center.first - width / 2, -center.second - NODE_HEIGHT / 2);
      } else {
        // Define points for a diamond-shaped node
        point1 = Vector2(center.first, -center.second + NODE_HEIGHT / 2);
        point2 = Vector2(center.first + width / 2, -center.second);
        point3 = Vector2(center.first, -center.second - NODE_HEIGHT / 2);
        point4 = Vector2(center.first - width / 2, -center.second);
      }

      firstPoint = point1;
      points.push_back(point1);
      points.push_back(point2);
      points.push_back(point3);
      points.push_back(point4);
      godotNode->set_polygon(points);
      nodeIdToGodoPos[node.first] = points;
    } else {
      // Code for an oval-shaped node
      int numPoints = 30; // Increase for smoother oval
      for (int i = 0; i < numPoints; ++i) {
        float angle = 2 * M_PI * i / numPoints;
        float x = center.first + width / 2 * cos(angle);
        float y = -center.second + NODE_HEIGHT / 2 * sin(angle);
        points.push_back(Vector2(x, y));
        if (i == 0)
          firstPoint = Vector2(x, y);
      }
      godotNode->set_polygon(points);

      // Define points for an oval bounding rectangle
      point1 = Vector2(center.first, -center.second + NODE_HEIGHT / 2);
      point2 = Vector2(center.first + width / 2, -center.second);
      point3 = Vector2(center.first, -center.second - NODE_HEIGHT / 2);
      point4 = Vector2(center.first - width / 2, -center.second);
      PackedVector2Array rectanglePoints;
      rectanglePoints.push_back(point1);
      rectanglePoints.push_back(point2);
      rectanglePoints.push_back(point3);
      rectanglePoints.push_back(point4);
      nodeIdToGodoPos[node.first] = rectanglePoints;
    }

    // Set the node color
    godotNode->set_color(colorNameToRGB.count(node.second.getColor())
                             ? colorNameToRGB.at(node.second.getColor())
                             : OPAQUE_WHITE);

    // Create and position the label
    Label *label = memnew(Label);
    label->set_text(node.second.getNodeId().c_str());
    label->add_theme_color_override("font_color", OPAQUE_BLACK);
    label->add_theme_font_size_override("font_size", 12);
    Vector2 size = label->get_combined_minimum_size();
    label->set_position(
        Vector2(center.first - size.x * 0.5, -(center.second + size.y * 0.5)));

    // Create a container for the label and the node
    CenterContainer *centerContainer = memnew(CenterContainer);
    centerContainer->set_size(Vector2(width, NODE_HEIGHT));
    centerContainer->set_position(
        Vector2(center.first - width / 2, -center.second - NODE_HEIGHT / 2));
    centerContainer->add_child(label);
    area2D->add_child(godotNode);
    area2D->add_child(centerContainer);
    nodeIdToPolygon[node.first] = godotNode;

    std::vector<Line2D *> lines;

    // Create the ouline of the node
    if (node.second.getDashed()) {
      points.push_back(firstPoint);
      createDashedLine(points, &lines, area2D);
    } else {
      outline->set_points(points);
      outline->add_point(firstPoint);
      lines.push_back(outline);
    }

    // Set outline color and width, and add to the area
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
    Vector2 previousPoint, lastPoint;
    std::vector<std::pair<float, float>> positions = edge.getPositions();
    PackedVector2Array linePoints;

    // Generate points for the edge line, inverting the y-axis due to a change
    // of reference in Godot.
    for (size_t i = 1; i < positions.size(); ++i) {
      Vector2 point = Vector2(positions.at(i).first, -positions.at(i).second);
      linePoints.push_back(point);
      previousPoint = lastPoint;
      lastPoint = point;
    }

    // Draw dashed or solid lines based on edge properties
    if (edge.getDashed()) {
      createDashedLine(linePoints, &lines, area2D);
    } else {
      Line2D *line = memnew(Line2D);
      line->set_points(linePoints);
      line->set_default_color(OPAQUE_BLACK);
      line->set_width(1);
      area2D->add_child(line);
      lines.push_back(line);
    }

    edgeIdToLines[edge.getEdgeId()] = lines;

    // Create and set up the arrowhead for the edge
    Polygon2D *arrowHead = memnew(Polygon2D);
    PackedVector2Array points;
    // Determine the orientation of the arrowhead
    if (previousPoint.x == lastPoint.x) {
      points.push_back(Vector2(lastPoint.x - 8, lastPoint.y));
      points.push_back(Vector2(lastPoint.x + 8, lastPoint.y));
      points.push_back(previousPoint.y < lastPoint.y
                           ? Vector2(lastPoint.x, lastPoint.y + 12)
                           : Vector2(lastPoint.x, lastPoint.y - 12));
    } else {
      points.push_back(Vector2(lastPoint.x, lastPoint.y + 8));
      points.push_back(Vector2(lastPoint.x, lastPoint.y - 8));
      points.push_back(previousPoint.x < lastPoint.x
                           ? Vector2(lastPoint.x + 12, lastPoint.y)
                           : Vector2(lastPoint.x - 12, lastPoint.y));
    }

    arrowHead->set_polygon(points);
    arrowHead->set_color(OPAQUE_BLACK);
    area2D->add_child(arrowHead);
    edgeIdToArrowHead[edge.getEdgeId()] = arrowHead;

    // Create a label for the edge
    Label *label = memnew(Label);
    label->set_text("");
    label->set_position(previousPoint);
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
      ChannelTransitions edgeStates = graph.getCycleEdgeStates().at(cycle);
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
