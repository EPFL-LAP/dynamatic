from generators.handshake.buffers.one_slot_break_dv import generate_one_slot_break_dv
from generators.support.utils import *


def generate_delay_buffer(name, params):
    latency = params[ATTR_LATENCY]

    if latency == 0:
        return _generate_no_lat_delay_buffer(name)
    elif latency == 1:
        return _generate_single_delay_buffer(name)
    else:
        return _generate_delay_buffer(name, latency)


def _generate_no_lat_delay_buffer(name):
    return f"""
MODULE {name}(ins_valid, outs_ready)

  -- output
  DEFINE ins_ready := outs_ready;
  DEFINE outs_valid := ins_valid;
"""


def _generate_single_delay_buffer(name):
    return f"""
MODULE {name}(ins_valid, outs_ready)
  VAR inner_one_slot_break_dv : {name}__one_slot_break_dv_dataless(ins, ins_valid, outs_ready);

  -- output
  DEFINE ins_ready := inner_one_slot_break_dv.ins_ready;
  DEFINE outs_valid := inner_one_slot_break_dv.outs_valid;

{generate_one_slot_break_dv(f"{name}__one_slot_break_dv_dataless", {ATTR_BITWIDTH: 0})}
"""


def _generate_delay_buffer(name, latency):
    no_one_slot_break_dv_latency = latency - 1
    return f"""
MODULE {name}(ins_valid, outs_ready)
  VAR inner_one_slot_break_dv : {name}__one_slot_break_dv_dataless(v{no_one_slot_break_dv_latency - 1}, outs_ready);
  {"\n  ".join([f"VAR v{n} : boolean;" for n in range(no_one_slot_break_dv_latency)])}

  ASSIGN init(v0) := FALSE;
  ASSIGN next(v0) := ins_valid;

  {"\n  ".join([f"""ASSIGN init(v{n + 1}) := FALSE;
  ASSIGN next(v{n + 1}) := inner_one_slot_break_dv.ins_ready ? v{n} : v{n + 1};
""" for n in range(no_one_slot_break_dv_latency - 1)])}

  -- output
  DEFINE ins_ready := inner_one_slot_break_dv.ins_ready;
  DEFINE outs_valid := inner_one_slot_break_dv.outs_valid;

{generate_one_slot_break_dv(f"{name}__one_slot_break_dv_dataless", {ATTR_BITWIDTH: 0})}
"""
