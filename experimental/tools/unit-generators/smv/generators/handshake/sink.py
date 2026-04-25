from generators.support.utils import *


def generate_sink(name, params):
    data_type = SmvScalarType(params[ATTR_BITWIDTH])

    if params["iog_terminator"]:
        if data_type.bitwidth == 0:
            return _generate_iog_terminator_dataless(name)
        else:
            return _generate_iog_terminator(name, data_type)
    else:
        if data_type.bitwidth == 0:
            return _generate_sink_dataless(name)
        else:
            return _generate_sink(name, data_type)


def _generate_iog_terminator_dataless(name):
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
  full_full := full;
"""

def _generate_iog_terminator(name, data_type):
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
  full_full := full;
"""

def _generate_sink_dataless(name):
    return f"""
MODULE {name}(ins_valid)

  -- output
  DEFINE
  ins_ready  :=  TRUE;
"""


def _generate_sink(name, data_type):
    return f"""
MODULE {name}(ins, ins_valid)

  -- output
  DEFINE
  ins_ready  :=  TRUE;
"""
