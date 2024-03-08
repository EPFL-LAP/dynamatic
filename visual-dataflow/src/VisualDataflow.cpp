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
#include "godot_cpp/classes/area2d.hpp"
#include "godot_cpp/classes/canvas_item.hpp"
#include "godot_cpp/classes/canvas_layer.hpp"
#include "godot_cpp/classes/center_container.hpp"
#include "godot_cpp/classes/color_rect.hpp"
#include "godot_cpp/classes/control.hpp"
#include "godot_cpp/classes/font.hpp"
#include "godot_cpp/classes/global_constants.hpp"
#include "godot_cpp/classes/label.hpp"
#include "godot_cpp/classes/line2d.hpp"
#include "godot_cpp/classes/node.hpp"
#include "godot_cpp/classes/node2d.hpp"
#include "godot_cpp/classes/panel.hpp"
#include "godot_cpp/classes/polygon2d.hpp"
#include "godot_cpp/classes/rich_text_label.hpp"
#include "godot_cpp/classes/style_box_flat.hpp"
#include "godot_cpp/classes/text_server.hpp"
#include "godot_cpp/core/class_db.hpp"
#include "godot_cpp/core/math.hpp"
#include "godot_cpp/core/memory.hpp"
#include "godot_cpp/variant/color.hpp"
#include "godot_cpp/variant/packed_vector2_array.hpp"
#include "godot_cpp/variant/utility_functions.hpp"
#include "godot_cpp/variant/vector2.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include <cstdint>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace godot;
using namespace dynamatic;
using namespace dynamatic::visual;

const godot::Color TRANSPARENT_BLACK(0, 0, 0, 0.075);
const godot::Color OPAQUE_BLACK(0, 0, 0, 1.0);
const godot::Color OPAQUE_WHITE(1, 1, 1, 1.0);

static const double LINE_WIDTH = 1.5;
static const double NODE_HEIGHT = 35;
static const double NODE_WIDTH_SCALING_COEFFICIENT = 70;
static const double DASH_LENGTH = 3;
static const double DASH_SPACE_LENGTH = DASH_LENGTH * 2;
static const unsigned NUM_OVAL_POINTS = 50;

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

    // Create the label and configure it
    RichTextLabel *bbLabel = memnew(RichTextLabel);
    bbLabel->set_use_bbcode(true);
    bbLabel->set_fit_content(true);
    bbLabel->set_autowrap_mode(TextServer::AUTOWRAP_OFF);
    bbLabel->set_position(
        Vector2(bb.boundries.at(0) + 5,
                -bb.labelPosition.second - bb.labelSize.first * 35));

    // Set the label's content
    bbLabel->push_font(get_theme_default_font(), 12);
    bbLabel->push_color(OPAQUE_BLACK);
    bbLabel->append_text(bb.label.c_str());
    bbLabel->pop();
    bbLabel->pop();

    add_child(bbLabel);
  }
}

static void setBasicLineProps(Line2D *line) {
  line->set_width(LINE_WIDTH);
  line->set_default_color(OPAQUE_BLACK);
  line->set_antialiased(true);
}

static void createDashedLine(PackedVector2Array &points,
                             std::vector<Line2D *> *lines, Area2D *area2D) {
  for (unsigned i = 0; i < points.size() - 1; ++i) {
    Vector2 start = points[i];
    Vector2 end = points[i + 1];
    Vector2 segment = end - start;
    double segmentLength = segment.length();
    segment = segment.normalized();
    double currentLength = 0.0;
    while (currentLength < segmentLength) {
      Line2D *line = memnew(Line2D);
      setBasicLineProps(line);
      Vector2 lineStart = start + segment * currentLength;
      double length = MIN(DASH_LENGTH, segmentLength - currentLength);
      Vector2 lineEnd = lineStart + segment * length;
      PackedVector2Array linePoints;
      linePoints.append(lineStart);
      linePoints.append(lineEnd);
      line->set_points(linePoints);

      lines->push_back(line);
      area2D->add_child(line);
      currentLength += DASH_LENGTH + DASH_SPACE_LENGTH;
    }
  }
}

void VisualDataflow::drawNodes() {

  for (auto &[id, node] : graph.getNodes()) {
    auto [centerX, centerY] = node.getPosition();
    float width = node.getWidth() * NODE_WIDTH_SCALING_COEFFICIENT;
    Area2D *area2D = memnew(Area2D);
    Polygon2D *godotNode = memnew(Polygon2D);
    Shape shape = node.getShape();

    double halfWidth = width / 2;
    double halfHeight = NODE_HEIGHT / 2;

    // Define the shape of the node as a sequence of 2D points
    PackedVector2Array points;
    if (shape == "diamond" || shape == "box") {

      if (shape == "box") {
        // Define points for a box-shaped node
        points.push_back(Vector2(centerX - halfWidth, -centerY + halfHeight));
        points.push_back(Vector2(centerX + halfWidth, -centerY + halfHeight));
        points.push_back(Vector2(centerX + halfWidth, -centerY - halfHeight));
        points.push_back(Vector2(centerX - halfWidth, -centerY - halfHeight));
      } else {
        // Define points for a diamond-shaped node
        points.push_back(Vector2(centerX, -centerY + halfHeight));
        points.push_back(Vector2(centerX + halfWidth, -centerY));
        points.push_back(Vector2(centerX, -centerY - halfHeight));
        points.push_back(Vector2(centerX - halfWidth, -centerY));
      }

      // The collision area is the same as the node itself
      nodeIdToGodoPos[id] = points;
    } else {
      // Create points to characterize an oval shape
      double angle = 0;
      double angleIncrement = 2 * M_PI / NUM_OVAL_POINTS;
      for (unsigned i = 0; i < NUM_OVAL_POINTS; ++i) {
        angle += angleIncrement;
        double x = centerX + width / 2 * cos(angle);
        double y = -centerY + NODE_HEIGHT / 2 * sin(angle);
        points.push_back(Vector2(x, y));
      }

      // The collison area is a rectangle centered within the oval
      PackedVector2Array rectanglePoints;
      rectanglePoints.push_back(Vector2(centerX, -centerY + NODE_HEIGHT / 2));
      rectanglePoints.push_back(Vector2(centerX + width / 2, -centerY));
      rectanglePoints.push_back(Vector2(centerX, -centerY - NODE_HEIGHT / 2));
      rectanglePoints.push_back(Vector2(centerX - width / 2, -centerY));
      nodeIdToGodoPos[id] = rectanglePoints;
    }

    // Sets the node's polygon and color
    godotNode->set_polygon(points);
    godotNode->set_color(colorNameToRGB.count(node.getColor())
                             ? colorNameToRGB.at(node.getColor())
                             : OPAQUE_WHITE);

    // Create the label and configure it
    RichTextLabel *nodeName = memnew(RichTextLabel);
    nodeName->set_use_bbcode(true);
    nodeName->set_fit_content(true);
    nodeName->set_autowrap_mode(TextServer::AUTOWRAP_OFF);
    nodeName->set_position(Vector2(centerX, -centerY));

    // Set the label's content
    nodeName->push_font(get_theme_default_font(), 11);
    nodeName->push_color(OPAQUE_BLACK);
    std::string text = "[center]" + node.getNodeId() + "[/center]";
    nodeName->append_text(text.c_str());
    nodeName->pop();
    nodeName->pop();

    // Create a container for the label and the node
    CenterContainer *centerContainer = memnew(CenterContainer);
    centerContainer->set_size(Vector2(width, NODE_HEIGHT));
    centerContainer->set_position(
        Vector2(centerX - halfWidth, -centerY - halfHeight));
    centerContainer->add_child(nodeName);
    area2D->add_child(godotNode);
    area2D->add_child(centerContainer);
    nodeIdToPolygon[id] = godotNode;

    // Create the ouline of the node
    Line2D *outline = memnew(Line2D);
    setBasicLineProps(outline);
    std::vector<Line2D *> lines;
    if (node.getDashed()) {
      points.push_back(points[0]);
      createDashedLine(points, &lines, area2D);
    } else {
      outline->set_points(points);
      outline->add_point(points[0]);
      lines.push_back(outline);
    }

    // Set outline color and width, and add to the area
    area2D->add_child(outline);
    add_child(area2D);
    nodeIdToContourLine[id] = lines;
    nodeIdToTransparency[id] = false;
  }
}

void VisualDataflow::drawEdges() {
  for (GraphEdge &edge : graph.getEdges()) {

    Area2D *area2D = memnew(Area2D);
    std::vector<Line2D *> lines;
    std::vector<std::pair<float, float>> positions = edge.getPositions();
    PackedVector2Array linePoints;

    // Generate points for the edge line, inverting the y-axis due to a change
    // of reference in Godot
    for (auto [x, y] : llvm::drop_begin(positions, 1))
      linePoints.push_back(Vector2(x, -y));

    // Draw dashed or solid lines based on edge properties
    if (edge.getDashed()) {
      createDashedLine(linePoints, &lines, area2D);
    } else {
      Line2D *line = memnew(Line2D);
      line->set_points(linePoints);
      setBasicLineProps(line);
      area2D->add_child(line);
      lines.push_back(line);
    }

    edgeIdToLines[edge.getEdgeId()] = lines;

    size_t numPoints = linePoints.size();
    Vector2 secondToLastPoint = linePoints[numPoints - 2];
    Vector2 lastPoint = linePoints[numPoints - 1];

    // Create and set up the arrowhead for the edge
    Polygon2D *arrowheadPoly = memnew(Polygon2D);
    arrowheadPoly->set_color(OPAQUE_BLACK);

    PackedVector2Array points;
    if (edge.getArrowhead() == "normal") {
      // Draw an arrow
      if (secondToLastPoint.x == lastPoint.x) {
        // Horizontal arrow
        points.push_back(Vector2(lastPoint.x - 5, lastPoint.y));
        points.push_back(Vector2(lastPoint.x + 5, lastPoint.y));
        points.push_back(secondToLastPoint.y < lastPoint.y
                             ? Vector2(lastPoint.x, lastPoint.y + 12)
                             : Vector2(lastPoint.x, lastPoint.y - 12));
      } else {
        // Vertical arrow
        points.push_back(Vector2(lastPoint.x, lastPoint.y + 5));
        points.push_back(Vector2(lastPoint.x, lastPoint.y - 5));
        points.push_back(secondToLastPoint.x < lastPoint.x
                             ? Vector2(lastPoint.x + 12, lastPoint.y)
                             : Vector2(lastPoint.x - 12, lastPoint.y));
      }
    } else {
      // Draw a circle
      double centerX = lastPoint.x, centerY = lastPoint.y, radius = 5;
      if (secondToLastPoint.x == lastPoint.x) {
        centerX = lastPoint.x;
        if (secondToLastPoint.y < lastPoint.y) {
          // Edge is going down
          centerY += radius;
        } else {
          // Edge is going up
          centerY -= radius;
        }
      } else {
        centerY = lastPoint.y;
        if (secondToLastPoint.x < lastPoint.x) {
          // Edge is going right
          centerX += radius;
        } else {
          // Edge is going left
          centerX -= radius;
        }
      }
      double angle = 0;
      double angleIncrement = 2 * M_PI / NUM_OVAL_POINTS;
      for (unsigned i = 0; i < NUM_OVAL_POINTS; ++i) {
        angle += angleIncrement;
        double x = centerX + radius * cos(angle);
        double y = centerY + radius * sin(angle);
        points.push_back(Vector2(x, y));
      }
    }

    arrowheadPoly->set_polygon(points);
    area2D->add_child(arrowheadPoly);
    edgeIdToArrowHead[edge.getEdgeId()] = arrowheadPoly;

    // Create the label and configure it
    RichTextLabel *label = memnew(RichTextLabel);
    label->set_use_bbcode(true);
    label->set_fit_content(true);
    label->set_autowrap_mode(TextServer::AUTOWRAP_OFF);
    // Lightly offset the label toward the bottom right compared to the start of
    // the line
    Vector2 firstPoint = linePoints[0];
    label->set_position(Vector2(firstPoint.x + 4, firstPoint.y + 4));

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
  if (cycle == cycleNb)
    return;

  int64_t maxCycle = (int64_t)cycleSlider->get_max();
  cycle = std::min(std::max(cycleNb, (int64_t)0), maxCycle);
  cycleLabel->set_text("Cycle: " + String::num_int64(cycle));
  cycleSlider->set_value(cycle);
  setEdgeColors(cycleNb);

  if (graph.getCycleEdgeStates().count(cycle)) {
    ChannelTransitions edgeStates = graph.getCycleEdgeStates().at(cycle);
    for (auto &[edgeID, state] : edgeStates) {
      std::vector<Line2D *> &lines = edgeIdToLines[edgeID];
      Polygon2D *arrowHead = edgeIdToArrowHead[edgeID];
      setEdgeColor(state.first, lines, arrowHead);

      // Display channel content if the valid wire is set
      RichTextLabel *dataValue = edgeIdToData.at(edgeID);
      dataValue->clear();
      if (state.first == State::STALL || state.first == State::TRANSFER) {
        // Write the data value only when it is valid
        dataValue->push_font(get_theme_default_font(), 11);
        dataValue->push_color(OPAQUE_BLACK);
        dataValue->append_text(state.second.c_str());
        dataValue->pop();
        dataValue->pop();
      }
    }
  }
}

void VisualDataflow::setEdgeColors(CycleNb cycle) {
  ChannelTransitions edgeStates = graph.getCycleEdgeStates().at(cycle);
  for (auto &[edgeID, state] : edgeStates) {
    std::vector<Line2D *> &lines = edgeIdToLines[edgeID];
    Polygon2D *arrowHead = edgeIdToArrowHead[edgeID];
    Color color = stateColors[state.first];
    for (Line2D *line : lines) {
      color.a = line->get_default_color().a;

      line->set_default_color(color);
    }
    arrowHead->set_color(color);
  }
}

void VisualDataflow::setEdgeColor(State state, std::vector<Line2D *> &lines,
                                  Polygon2D *arrowHead) {
  Color color = stateColors[state];
  for (Line2D *line : lines) {
    color.a = line->get_default_color().a;
    line->set_default_color(color);
  }
  arrowHead->set_color(color);
}

void VisualDataflow::changeStateColor(int64_t state, Color color) {
  // Update the color associated to the state
  State stateEnum;
  if (state == 0)
    stateEnum = State::UNDEFINED;
  else if (state == 1)
    stateEnum = State::IDLE;
  else if (state == 2)
    stateEnum = State::ACCEPT;
  else if (state == 3)
    stateEnum = State::STALL;
  else if (state == 4)
    stateEnum = State::TRANSFER;
  else
    llvm_unreachable("invalid channel state!");
  stateColors[stateEnum] = color;

  // Change color of all edges currently in the state whose color was changed
  CycleTransitions &cycleStates = graph.getCycleEdgeStates();
  for (auto &[edgeID, state] : cycleStates[cycle]) {
    if (state.first == stateEnum) {
      std::vector<Line2D *> &lines = edgeIdToLines[edgeID];
      Polygon2D *arrowHead = edgeIdToArrowHead[edgeID];
      setEdgeColor(stateEnum, lines, arrowHead);
    }
  }
}

static double crossProduct(Vector2 a, Vector2 b, Vector2 c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

static bool isInside(Vector2 p, Vector2 a, Vector2 b, Vector2 c, Vector2 d) {
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
