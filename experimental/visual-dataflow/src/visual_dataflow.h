#ifndef VISUAL_DATAFLOW_VISUAL_DATAFLOW_H
#define VISUAL_DATAFLOW_VISUAL_DATAFLOW_H

#include "godot_cpp/classes/sprite2d.hpp"

namespace godot {

class VisualDataflow : public Sprite2D {
  GDCLASS(VisualDataflow, Sprite2D)

private:
  double timePassed;

protected:
  static void _bind_methods();

public:
  VisualDataflow();

  unsigned getNumber();

  ~VisualDataflow() override = default;

  void _process(double delta) override;
};

} // namespace godot

#endif // VISUAL_DATAFLOW_VISUAL_DATAFLOW_H
