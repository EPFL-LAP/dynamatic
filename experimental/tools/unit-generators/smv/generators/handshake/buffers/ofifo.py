from generators.support.utils import *
from generators.handshake.buffers.fifo_break_dv import generate_fifo_break_dv
from generators.handshake.buffers.one_slot_break_r import generate_one_slot_break_r
from generators.support.buffer_counter import generate_buffer_counter


def generate_ofifo(name, params):
    slots = params[ATTR_SLOTS]
    data_type = SmvScalarType(params[ATTR_BITWIDTH])
    debug_counter = params.get(ATTR_DEBUG_COUNTER, False)

    if data_type.bitwidth == 0:
        return _generate_ofifo_dataless(name, slots, debug_counter)
    else:
        return _generate_ofifo(name, slots, data_type, debug_counter)


def _generate_ofifo_dataless(name, slots, debug_counter):
    return f"""
MODULE {name} (ins_valid, outs_ready)
  VAR
  inner_tehb : {name}__tehb_dataless(ins_valid, inner_elastic_fifo.ins_ready);
  inner_elastic_fifo : {name}__elastic_fifo_inner_dataless(inner_tehb.outs_valid, outs_ready);
  {f"debug_counter : {name}__debug_counter(ins_valid, ins_ready, outs_valid, outs_ready);" if debug_counter else ""}

  -- output
  DEFINE
  ins_ready := inner_tehb.ins_ready;
  outs_valid := inner_elastic_fifo.outs_valid;

{generate_one_slot_break_r(f"{name}__tehb_dataless", {ATTR_BITWIDTH: 0})}
{generate_fifo_break_dv(f"{name}__elastic_fifo_inner_dataless", {ATTR_SLOTS: slots, ATTR_BITWIDTH: 0})}
{generate_buffer_counter(f"{name}__debug_counter", slots) if debug_counter else ""}
"""


def _generate_ofifo(name, slots, data_type, debug_counter):
    return f"""
MODULE {name} (ins, ins_valid, outs_ready)
  VAR
  inner_tehb : {name}__tehb(ins, ins_valid, inner_elastic_fifo.ins_ready);
  inner_elastic_fifo : {name}__elastic_fifo_inner(inner_tehb.outs, inner_tehb.outs_valid, outs_ready);
  {f"debug_counter : {name}__debug_counter(ins_valid, ins_ready, outs_valid, outs_ready);" if debug_counter else ""}

  -- output
  DEFINE
  ins_ready := inner_tehb.ins_ready;
  outs_valid := inner_elastic_fifo.outs_valid;
  outs := inner_elastic_fifo.outs;

{generate_one_slot_break_r(f"{name}__tehb", {ATTR_BITWIDTH: data_type.bitwidth})}
{generate_fifo_break_dv(f"{name}__elastic_fifo_inner", {ATTR_SLOTS: slots, ATTR_BITWIDTH: data_type.bitwidth})}
{generate_buffer_counter(f"{name}__debug_counter", slots) if debug_counter else ""}
"""
