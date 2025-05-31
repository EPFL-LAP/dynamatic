from generators.support.utils import *


def generate_constant(name, params):
    value = params[ATTR_VALUE]
    data_type = SmvScalarType(params[ATTR_BITWIDTH])

    return _generate_constant(name, value, data_type)


def _generate_constant(name, value, data_type):
    return f"""
MODULE {name}(ctrl_valid, outs_ready)

  -- output
  DEFINE
  ctrl_ready := outs_ready;
  outs_valid := ctrl_valid;
  outs := {data_type.format_constant(value)};
"""
