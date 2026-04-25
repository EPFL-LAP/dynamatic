from generators.support.utils import *


def generate_sink(name, params):
    data_type = SmvScalarType(params[ATTR_BITWIDTH])

    if "pyTest" in params:
        return _generate_iog_terminating_sink_dataless(name)
    if data_type.bitwidth == 0:
        return _generate_sink_dataless(name)
    else:
        return _generate_sink(name, data_type)


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


def _generate_iog_terminating_sink_dataless(name):
    return f"""
MODULE {name}(ins_valid)

  VAR
  slot_full : bool;
  -- output
  DEFINE
  ins_ready  :=  !slot_full;
"""
