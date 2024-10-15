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
#include "dynamatic/Support/DOT.h"
#include "godot_cpp/classes/area2d.hpp"
#include "godot_cpp/classes/canvas_item.hpp"
#include "godot_cpp/classes/center_container.hpp"
#include "godot_cpp/classes/font.hpp"
#include "godot_cpp/classes/label.hpp"
#include "godot_cpp/classes/line2d.hpp"
#include "godot_cpp/classes/polygon2d.hpp"
#include "godot_cpp/classes/rich_text_label.hpp"
#include "godot_cpp/classes/text_server.hpp"
#include "godot_cpp/core/class_db.hpp"
#include "godot_cpp/core/math.hpp"
#include "godot_cpp/core/memory.hpp"
#include "godot_cpp/variant/color.hpp"
#include "godot_cpp/variant/packed_vector2_array.hpp"
#include "godot_cpp/variant/vector2.hpp"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace godot;
using namespace dynamatic;
using namespace dynamatic::visual;

const godot::Color TRANSPARENT_BLACK(0, 0, 0, 0.075);
const godot::Color OPAQUE_BLACK(0, 0, 0, 1.0);
const godot::Color OPAQUE_WHITE(1, 1, 1, 1.0);

static constexpr double LINE_WIDTH = 1.5, NODE_HEIGHT = 35,
                        NODE_WIDTH_SCALING_COEFFICIENT = 70, DASH_LENGTH = 3,
                        DASH_SPACE_LENGTH = DASH_LENGTH * 2;
static constexpr unsigned NUM_OVAL_POINTS = 50;

static constexpr double SELECT_TRANSPARENCY = 1.0, UNSELECT_TRANSPARENCY = 0.3;

/// Maps color names to their corresponding RGBA Color values in Godot
static const std::unordered_map<std::string, Color> NAMED_COLORS = {
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

static void setTransparency(Polygon2D *polygon, double transparency) {
  Color c = polygon->get_color();
  c.a = transparency;
  polygon->set_color(c);
}

static void setTransparency(Line2D *line, double transparency) {
  Color c = line->get_default_color();
  c.a = transparency;
  line->set_default_color(c);
}

void VisualDataflow::_bind_methods() {

  ClassDB::bind_method(D_METHOD("start", "inputDOTFile", "inputCSVFile"),
                       &VisualDataflow::start);
  ClassDB::bind_method(D_METHOD("nextCycle"), &VisualDataflow::nextCycle);
  ClassDB::bind_method(D_METHOD("previousCycle"),
                       &VisualDataflow::previousCycle);
  ClassDB::bind_method(D_METHOD("changeCycle", "cycleNb"),
                       &VisualDataflow::gotoCycle);
  ClassDB::bind_method(D_METHOD("changeStateColor", "state", "color"),
                       &VisualDataflow::changeStateColor);
  ClassDB::bind_method(D_METHOD("onClick", "position"),
                       &VisualDataflow::onClick);
  ClassDB::bind_method(D_METHOD("resetSelection"),
                       &VisualDataflow::resetSelection);
}

void VisualDataflow::start(const godot::String &dotFilepath,
                           const godot::String &csvFilepath) {
  cycleLabel = (Label *)get_node_internal(
      NodePath("CanvasLayer/Timeline/MarginContainer/VBoxContainer/"
               "HBoxContainer/CycleNumber"));
  cycleSlider = (HSlider *)get_node_internal(
      NodePath("CanvasLayer/Timeline/MarginContainer/VBoxContainer/HSlider"));

  if (failed(graph.fromDOTAndCSV(dotFilepath.utf8().get_data(),
                                 csvFilepath.utf8().get_data()))) {
    llvm::errs() << "Failed to parse graph data\n";
    return;
  }
  maxCycle = graph.getLastCycleIdx();
  cycleSlider->set_max(maxCycle);
  drawGraph();
}

void VisualDataflow::drawBBs() {
  for (const DOTGraph::Subgraph &subgraph : graph.getGraph().getSubgraphs()) {
    const GodotGraph::SubgraphProps &props =
        graph.getSubgraphProperties(&subgraph);

    std::vector<float> boundaries = props.boundaries;
    Polygon2D *p = memnew(Polygon2D);
    PackedVector2Array points;
    points.push_back(Vector2(boundaries.at(0), -boundaries.at(1)));
    points.push_back(Vector2(boundaries.at(2), -boundaries.at(1)));
    points.push_back(Vector2(boundaries.at(2), -boundaries.at(3)));
    points.push_back(Vector2(boundaries.at(0), -boundaries.at(3)));

    p->set_polygon(points);
    p->set_color(TRANSPARENT_BLACK);

    add_child(p);

    // Create the label and configure it
    RichTextLabel *bbLabel = memnew(RichTextLabel);
    bbLabel->set_use_bbcode(true);
    bbLabel->set_fit_content(true);
    bbLabel->set_autowrap_mode(TextServer::AUTOWRAP_OFF);
    bbLabel->set_position(
        Vector2(props.boundaries.at(0) + 5,
                -props.labelPosition.second - props.labelSize.first * 35));

    // Set the label's content
    bbLabel->push_font(get_theme_default_font(), 12);
    bbLabel->push_color(OPAQUE_BLACK);
    bbLabel->append_text(props.label.c_str());
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

  for (const DOTGraph::Node *node : graph.getGraph().getNodes()) {
    const GodotGraph::NodeProps &props = graph.getNodeProperties(node);
    NodeGeometry &geometry = nodeGeometries.try_emplace(node).first->second;

    auto [centerX, centerY] = props.position;
    float width = props.width * NODE_WIDTH_SCALING_COEFFICIENT;
    Area2D *area2D = memnew(Area2D);
    Polygon2D *godotNode = memnew(Polygon2D);
    std::string shape = props.shape;

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
      geometry.collision = points;
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
      geometry.collision = rectanglePoints;
    }

    // Sets the node's polygon and color
    godotNode->set_polygon(points);
    godotNode->set_color(NAMED_COLORS.count(props.color)
                             ? NAMED_COLORS.at(props.color)
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
    std::string text = "[center]" + node->id + "[/center]";
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
    geometry.shape = godotNode;

    // Create the ouline of the node
    Line2D *outline = memnew(Line2D);
    setBasicLineProps(outline);
    std::vector<Line2D *> lines;
    if (props.isDotted) {
      points.push_back(points[0]);
      createDashedLine(points, &geometry.shapeLines, area2D);
    } else {
      outline->set_points(points);
      outline->add_point(points[0]);
      geometry.shapeLines.push_back(outline);
    }
    // Set outline color and width, and add to the area
    area2D->add_child(outline);
    add_child(area2D);
  }
}

void VisualDataflow::drawEdges() {
  for (const DOTGraph::Edge *edge : graph.getGraph().getEdges()) {
    const GodotGraph::EdgeProps &props = graph.getEdgeProperties(edge);
    EdgeGeometry &geometry = edgeGeometries.try_emplace(edge).first->second;

    Area2D *area2D = memnew(Area2D);
    std::vector<std::pair<float, float>> positions = props.positions;
    PackedVector2Array linePoints;

    // Generate points for the edge line, inverting the y-axis due to a change
    // of reference in Godot
    for (auto [x, y] : llvm::drop_begin(positions, 1))
      linePoints.push_back(Vector2(x, -y));

    // Draw dashed or solid lines based on edge properties
    if (props.isDotted) {
      createDashedLine(linePoints, &geometry.segments, area2D);
    } else {
      Line2D *line = memnew(Line2D);
      line->set_points(linePoints);
      setBasicLineProps(line);
      area2D->add_child(line);
      geometry.segments.push_back(line);
    }

    size_t numPoints = linePoints.size();
    Vector2 secondToLastPoint = linePoints[numPoints - 2];
    Vector2 lastPoint = linePoints[numPoints - 1];

    // Create and set up the arrowhead for the edge
    Polygon2D *arrowheadPoly = memnew(Polygon2D);
    arrowheadPoly->set_color(OPAQUE_BLACK);

    PackedVector2Array points;
    if (props.arrowhead == "normal") {
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
    geometry.arrowhead = arrowheadPoly;

    // Create the label and configure it
    geometry.data = memnew(RichTextLabel);
    geometry.data->set_use_bbcode(true);
    geometry.data->set_fit_content(true);
    geometry.data->set_autowrap_mode(TextServer::AUTOWRAP_OFF);
    // Lightly offset the label toward the bottom right compared to the start of
    // the line
    Vector2 firstPoint = linePoints[0];
    geometry.data->set_position(Vector2(firstPoint.x + 4, firstPoint.y + 4));

    add_child(geometry.data);
    add_child(area2D);
  }
}

void VisualDataflow::drawGraph() {
  drawBBs();
  drawNodes();
  drawEdges();
  drawCycle();
}

void VisualDataflow::nextCycle() {
  if (cycle != maxCycle) {
    ++cycle;
    drawCycle();
  }
}

void VisualDataflow::previousCycle() {
  if (cycle != 0) {
    --cycle;
    drawCycle();
  }
}

void VisualDataflow::gotoCycle(int64_t cycleNum) {
  unsigned saveCycle = cycle;
  cycle = cycleNum < 0 ? 0 : std::min(maxCycle, (unsigned)cycleNum);
  if (saveCycle != cycle)
    drawCycle();
}

void VisualDataflow::drawCycle() {
  // Edge case where there are no cycles
  if (maxCycle == 0)
    return;

  cycleLabel->set_text("Cycle: " + String::num_uint64(cycle));
  cycleSlider->set_value(cycle);

  for (const auto &[edge, edgeState] : graph.getChanges(cycle)) {
    assert(edgeGeometries.contains(edge) && "unknown edge");
    EdgeGeometry &geo = edgeGeometries.getOrInsertDefault(edge);
    const Color &color = stateColors.at(edgeState.state);
    geo.setColor(color);

    // Display channel content if the valid wire is set
    geo.data->clear();
    if (edgeState.state == DataflowState::STALL ||
        edgeState.state == DataflowState::TRANSFER) {
      // Write the data value only when it is valid
      geo.data->push_font(get_theme_default_font(), 11);
      geo.data->push_color(OPAQUE_BLACK);
      geo.data->append_text(edgeState.data.c_str());
      geo.data->pop();
      geo.data->pop();
    }
  }
}

void VisualDataflow::changeStateColor(int64_t state, Color color) {
  static const std::array<DataflowState, 5> intToState = {
      DataflowState::UNDEFINED, DataflowState::IDLE, DataflowState::ACCEPT,
      DataflowState::STALL, DataflowState::TRANSFER};

  // Update the color associated to the state
  assert(state < 0 || state > 4 && "invalid channel state!");
  DataflowState stateEnum = intToState[state];
  stateColors.insert_or_assign(stateEnum, color);

  // Change color of all edges currently in the state whose color was changed
  for (const auto &[edge, edgeState] : graph.getChanges(cycle)) {
    if (edgeState.state == stateEnum) {
      Color &color = stateColors[edgeState.state];
      assert(edgeGeometries.contains(edge) && "unknown edge");
      edgeGeometries.getOrInsertDefault(edge).setColor(color);
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
  for (auto &[node, geometry] : nodeGeometries) {
    PackedVector2Array &points = geometry.collision;
    if (!isInside(position, points[0], points[1], points[2], points[3]))
      continue;

    if (selectedNodes.contains(node)) {
      // Node was already selected, remove it from the selection set
      selectedNodes.erase(node);

      if (selectedNodes.empty()) {
        // Nothing is currently selected, everything should become opaque again
        setGraphTransparency(SELECT_TRANSPARENCY);
        selectedEdges.clear();
        return;
      }

      // Update the selection set of adjacent edges; those whose other endpoint
      // is selected become (single-)selected, others become unselected
      nodeGeometries.getOrInsertDefault(node).setTransparency(
          UNSELECT_TRANSPARENCY);
      SmallPtrSet<const DOTGraph::Edge *, 4> edgesToUnselect;
      for (const auto *edge : graph.getGraph().getAdjacentEdges(*node)) {
        auto it = selectedEdges.find(edge);
        assert(it != selectedEdges.end() && "adjacent edge not selected");
        bool &doubleSelected = it->getSecond();
        if (doubleSelected) {
          doubleSelected = false;
        } else {
          edgesToUnselect.insert(edge);
          edgeGeometries.getOrInsertDefault(edge).setTransparency(
              UNSELECT_TRANSPARENCY);
        }
      }
      llvm::for_each(edgesToUnselect,
                     [&](const auto *e) { selectedEdges.erase(e); });

    } else {
      // Node was not selected, add it to the selection set

      if (selectedNodes.empty()) {
        // Nothing is currently selected, everything should become transparent
        // first
        setGraphTransparency(UNSELECT_TRANSPARENCY);
      }
      selectedNodes.insert(node);

      // Update the selection set of adjacent edges; those whose other endpoint
      // is selected become (double-)selected, others become selected
      nodeGeometries.getOrInsertDefault(node).setTransparency(
          SELECT_TRANSPARENCY);
      for (const auto *edge : graph.getGraph().getAdjacentEdges(*node)) {
        if (auto it = selectedEdges.find(edge); it != selectedEdges.end()) {
          // Edge already selected, just mark it as selected "twice"
          it->getSecond() = true;
        } else {
          selectedEdges.insert({edge, false});
          edgeGeometries.getOrInsertDefault(edge).setTransparency(
              SELECT_TRANSPARENCY);
        }
      }
    }
    return;
  }
}

void VisualDataflow::setGraphTransparency(double transparency) {
  for (auto &[_, geo] : nodeGeometries)
    geo.setTransparency(transparency);
  for (auto &[_, geo] : edgeGeometries)
    geo.setTransparency(transparency);
}

void VisualDataflow::NodeGeometry::setTransparency(double transparency) {
  ::setTransparency(shape, transparency);
  for (Line2D *line : shapeLines)
    ::setTransparency(line, transparency);
}

void VisualDataflow::EdgeGeometry::setTransparency(double transparency) {
  ::setTransparency(arrowhead, transparency);
  for (Line2D *seg : segments)
    ::setTransparency(seg, transparency);
}

void VisualDataflow::EdgeGeometry::setColor(Color color) {
  color.a = arrowhead->get_color().a;
  for (Line2D *seg : segments)
    seg->set_default_color(color);
  arrowhead->set_color(color);
}

void VisualDataflow::resetSelection() {
  selectedNodes.clear();
  selectedEdges.clear();
  setGraphTransparency(1);
}
