from generators.support.utils import *


def generate_init(name, params):
    data_type = SmvScalarType(params[ATTR_BITWIDTH])
    # the value of the token the init holds on reset
    initial_value = params.get("initial_value", 0)

    if data_type.bitwidth == 0:
        return _generate_init_dataless(name)
    else:
        return _generate_init(name, data_type, initial_value)


# a one slot buffer breaking ready, full on reset
def _generate_init_dataless(name):
    return f"""
MODULE {name}(ins_valid, outs_ready)
  VAR
  full : boolean;

  ASSIGN
  init(full) := TRUE;
  next(full) := outs_valid & !outs_ready;

  -- output
  DEFINE
  ins_ready := !full;
  outs_valid := ins_valid | full;
"""


# the same, holding the initial value on reset
def _generate_init(name, data_type, initial_value):
    return f"""
MODULE {name}(ins, ins_valid, outs_ready)
  VAR
  inner_init : {name}__init_dataless(ins_valid, outs_ready);
  data : {data_type};

  ASSIGN
  init(data) := {data_type.format_constant(initial_value)};
  next(data) := ins_ready & ins_valid & !outs_ready ? ins : data;

  -- output
  DEFINE
  ins_ready := inner_init.ins_ready;
  outs_valid := inner_init.outs_valid;
  outs := inner_init.full ? data : ins;

  full := inner_init.full;

{_generate_init_dataless(f"{name}__init_dataless")}
"""
