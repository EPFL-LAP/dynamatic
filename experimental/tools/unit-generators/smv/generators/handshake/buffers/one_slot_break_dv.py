from generators.support.utils import *
from generators.support.buffer_counter import generate_buffer_counter


def generate_one_slot_break_dv(name, params):
    data_type = SmvScalarType(params[ATTR_BITWIDTH])

    if data_type.bitwidth == 0:
        return _generate_one_slot_break_dv_dataless(name)
    else:
        return _generate_one_slot_break_dv(name, data_type)


def _generate_one_slot_break_dv_dataless(name):
    counter_name = f"{name}__counter"
    return f"""
MODULE {name} (ins_valid, outs_ready)
  VAR
  outs_valid_i : boolean;
  inner_counter : {counter_name}(ins_valid, ins_ready, outs_valid, outs_ready);

  ASSIGN
  init(outs_valid_i) := FALSE;
  next(outs_valid_i) := ins_valid | (outs_valid_i & !outs_ready);

  -- output
  DEFINE
  ins_ready := !outs_valid_i | outs_ready;
  outs_valid := outs_valid_i;

{generate_buffer_counter(counter_name, 1)}
"""


def _generate_one_slot_break_dv(name, data_type):
    counter_name = f"{name}__counter"
    return f"""
MODULE {name} (ins, ins_valid, outs_ready)
  VAR
  inner_one_slot_break_dv : {name}__one_slot_break_dv_dataless(ins_valid, outs_ready);
  data : {data_type};
  inner_counter : {counter_name}(ins_valid, ins_ready, outs_valid, outs_ready);

  ASSIGN
  init(data) := {data_type.format_constant(0)};
  next(data) := case
    ins_ready & ins_valid : ins;
    TRUE : data;
  esac;
    
  -- output
  DEFINE
  ins_ready := inner_one_slot_break_dv.ins_ready;
  outs_valid := inner_one_slot_break_dv.outs_valid;
  outs := data;

{_generate_one_slot_break_dv_dataless(f"{name}__one_slot_break_dv_dataless")}
{generate_buffer_counter(counter_name, 1)}
"""
