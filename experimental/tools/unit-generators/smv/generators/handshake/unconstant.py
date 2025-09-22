from generators.support.utils import *


def generate_unconstant(name, params):
    data_type = SmvScalarType(params[ATTR_BITWIDTH])

    return _generate_unconstant(name, data_type)


def _generate_unconstant(name, _):
    return f"""
MODULE {name}(data, data_valid, ctrl_ready)

  -- output
  DEFINE
  data_ready := ctrl_ready;
  ctrl_valid := data_valid;
"""
