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
#include "godot_cpp/classes/label.hpp"
#include "godot_cpp/classes/node.hpp"
#include "godot_cpp/classes/panel.hpp"
#include "godot_cpp/core/class_db.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"

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
  GraphParser parser =
      GraphParser("/home/qgross/Documents/dynamatic/experimental/"
                  "visual-dataflow/test/bicg.dot");
  if (failed(parser.parse(&graph))) {
    return;
  }

  Label graph_label = Label();
  graph_label.set_text("My first graph");
  add_child(&graph_label);

  size_t nodeCounter = 0;

  for (auto &node : graph.getNodes()) {
    nodeCounter++;
    Panel *panel = memnew(Panel);
    panel->set_custom_minimum_size(Vector2(200, 100));
    panel->set_position(Vector2(nodeCounter * 100, nodeCounter * 100));
    Label node_label = Label();
    node_label.set_text(node.second.getNodeId().c_str());
    panel->add_child(&node_label);
    add_child(panel);
  }
}
