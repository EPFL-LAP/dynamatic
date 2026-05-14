from generators.support.utils import *


def generate_dead_buffer(name, params):
    data_type = SmvScalarType(params[ATTR_BITWIDTH])

    if data_type.bitwidth == 0:
        return _generate_dead_buffer_dataless(name)
    else:
        return _generate_dead_buffer(name, data_type)


def _generate_dead_buffer_dataless(name):
    return f"""
MODULE {name}(ins_valid)
  VAR
  full : boolean;

  ASSIGN
  init(full) := FALSE;
  next(full) := full | ins_valid;

  -- output
  DEFINE
  ins_ready  :=  !full;
"""


def _generate_dead_buffer(name, data_type):
    return f"""
MODULE {name}(ins, ins_valid)
  VAR
  full : boolean;

  ASSIGN
  init(full) := FALSE;
  next(full) := full | ins_valid;

  -- output
  DEFINE
  ins_ready  :=  !full;
"""
