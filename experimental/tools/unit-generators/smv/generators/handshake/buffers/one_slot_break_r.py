from generators.support.utils import *
from generators.support.buffer_counter import generate_buffer_counter


def generate_one_slot_break_r(name, params):
    data_type = SmvScalarType(params[ATTR_BITWIDTH])
    debug_counter = params.get(ATTR_DEBUG_COUNTER, False)

    if data_type.bitwidth == 0:
        return _generate_one_slot_break_r_dataless(name, debug_counter)
    else:
        return _generate_one_slot_break_r(name, data_type, debug_counter)


def _generate_one_slot_break_r_dataless(name, debug_counter):
    return f"""
MODULE {name}(ins_valid, outs_ready)
  VAR
  full : boolean;
  {f"debug_counter : {name}__debug_counter(ins_valid, ins_ready, outs_valid, outs_ready);" if debug_counter else ""}

  ASSIGN
  init(full) := FALSE;
  next(full) := outs_valid & !outs_ready;

  -- output
  DEFINE
  ins_ready := !full;
  outs_valid := ins_valid | full;
{generate_buffer_counter(f"{name}__debug_counter", 1) if debug_counter else ""}
"""


def _generate_one_slot_break_r(name, data_type, debug_counter):
    return f"""
MODULE {name}(ins, ins_valid, outs_ready)
  VAR
  inner_one_slot_break_r : {name}__one_slot_break_r_dataless(ins_valid, outs_ready);
  data : {data_type};
  {f"debug_counter : {name}__debug_counter(ins_valid, ins_ready, outs_valid, outs_ready);" if debug_counter else ""}

  ASSIGN
  init(data) := {data_type.format_constant(0)};
  next(data) := ins_ready & ins_valid & !outs_ready ? ins : data;

  -- output
  DEFINE
  ins_ready := inner_one_slot_break_r.ins_ready;
  outs_valid := inner_one_slot_break_r.outs_valid;
  outs := inner_one_slot_break_r.full ? data : ins;

{_generate_one_slot_break_r_dataless(f"{name}__one_slot_break_r_dataless", False)}
{generate_buffer_counter(f"{name}__debug_counter", 1) if debug_counter else ""}
"""
