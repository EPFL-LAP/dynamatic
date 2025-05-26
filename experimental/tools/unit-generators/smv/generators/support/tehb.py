from generators.support.utils import *


def generate_tehb(name, params):
    data_type = SmvScalarType(params[ATTR_BITWIDTH])

    if data_type.bitwidth == 0:
        return _generate_tehb_dataless(name)
    else:
        return _generate_tehb(name, data_type)


def _generate_tehb_dataless(name):
    return f"""
MODULE {name}(ins_valid, outs_ready)
  VAR
  full : boolean;

  ASSIGN
  init(full) := FALSE;
  next(full) := outs_valid & !outs_ready;

  -- output
  DEFINE
  ins_ready := !full;
  outs_valid := ins_valid | full;
"""


def _generate_tehb(name, data_type):
    return f"""
MODULE {name}(ins, ins_valid, outs_ready)
  VAR
  inner_tehb : {name}__tehb_dataless(ins_valid, outs_ready);
  data : {data_type};

  ASSIGN
  init(data) := {data_type.format_constant(0)};
  next(data) := ins_ready & ins_valid & !outs_ready ? ins : data;

  -- output
  DEFINE
  ins_ready := inner_tehb.ins_ready;
  outs_valid := inner_tehb.outs_valid;
  outs := inner_tehb.full ? data : ins;

{_generate_tehb_dataless(f"{name}__tehb_dataless")}
"""
