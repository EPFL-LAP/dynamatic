from generators.support.utils import *


def generate_one_slot_break_r(name, params):
    data_type = SmvScalarType(params[ATTR_BITWIDTH])

    if data_type.bitwidth == 0:
        return _generate_one_slot_break_r_dataless(name)
    else:
        return _generate_one_slot_break_r(name, data_type)


def _generate_one_slot_break_r_dataless(name):
    return f"""
MODULE {name}(ins_valid, outs_ready)
  VAR
  full_i : boolean;

  ASSIGN
  init(full_i) := FALSE;
  next(full_i) := outs_valid & !outs_ready;

  -- output
  DEFINE
  ins_ready := !full_i;
  outs_valid := ins_valid | full_i;
  full := full_i;
"""


def _generate_one_slot_break_r(name, data_type):
    return f"""
MODULE {name}(ins, ins_valid, outs_ready)
  VAR
  inner_one_slot_break_r : {name}__one_slot_break_r_dataless(ins_valid, outs_ready);
  data : {data_type};

  ASSIGN
  init(data) := {data_type.format_constant(0)};
  next(data) := ins_ready & ins_valid & !outs_ready ? ins : data;

  -- output
  DEFINE
  ins_ready := inner_one_slot_break_r.ins_ready;
  outs_valid := inner_one_slot_break_r.outs_valid;
  outs := inner_one_slot_break_r.full ? data : ins;

{_generate_one_slot_break_r_dataless(f"{name}__one_slot_break_r_dataless")}
"""
