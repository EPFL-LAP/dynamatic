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

#ifndef VISUAL_DATAFLOW_VISUAL_DATAFLOW_H
#define VISUAL_DATAFLOW_VISUAL_DATAFLOW_H

#include "godot_cpp/classes/control.hpp"
#include <vector>

namespace godot {

class VisualDataflow : public Control {
  GDCLASS(VisualDataflow, Control)

private:
  struct Node {
    double x;
    double y;
  };

  int numberOfNodes = 3;
  // std::vector<Node> nodes;

protected:
  // NOLINTNEXTLINE(readability-identifier-naming)
  static void _bind_methods();

public:
  VisualDataflow();

  ~VisualDataflow() override = default;

  void myProcess(double delta);

  double getNodePosX(int index);
  double getNodePosY(int index);

  int getNumberOfNodes();
};

} // namespace godot

#endif // VISUAL_DATAFLOW_VISUAL_DATAFLOW_H
