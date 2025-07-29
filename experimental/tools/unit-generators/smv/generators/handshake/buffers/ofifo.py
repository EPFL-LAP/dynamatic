from generators.support.utils import *
from generators.handshake.buffers.fifo_break_dv import generate_fifo_break_dv
from generators.handshake.buffers.one_slot_break_r import generate_one_slot_break_r


def generate_ofifo(name, params):
    slots = params[ATTR_SLOTS]
    data_type = SmvScalarType(params[ATTR_BITWIDTH])

    if data_type.bitwidth == 0:
        return _generate_ofifo_dataless(name, slots)
    else:
        return _generate_ofifo(name, slots, data_type)


def _generate_ofifo_dataless(name, slots):
    return f"""
MODULE {name} (ins_valid, outs_ready)
  VAR
  inner_tehb : {name}__tehb_dataless(ins_valid, inner_elastic_fifo.ins_ready);
  inner_elastic_fifo : {name}__elastic_fifo_inner_dataless(inner_tehb.outs_valid, outs_ready);

  -- output
  DEFINE
  ins_ready := inner_tehb.ins_ready;
  outs_valid := inner_elastic_fifo.outs_valid;

{generate_one_slot_break_r(f"{name}__tehb_dataless", {ATTR_BITWIDTH: 0})}
{generate_fifo_break_dv(f"{name}__elastic_fifo_inner_dataless", {ATTR_SLOTS: slots, ATTR_BITWIDTH: 0})}
"""


def _generate_ofifo(name, slots, data_type):
    return f"""
MODULE {name} (ins, ins_valid, outs_ready)
  VAR
  inner_tehb : {name}__tehb(ins, ins_valid, inner_elastic_fifo.ins_ready);
  inner_elastic_fifo : {name}__elastic_fifo_inner(inner_tehb.outs, inner_tehb.outs_valid, outs_ready);

  -- output
  DEFINE
  ins_ready := inner_tehb.ins_ready;
  outs_valid := inner_elastic_fifo.outs_valid;
  outs := inner_elastic_fifo.outs;

{generate_one_slot_break_r(f"{name}__tehb", {ATTR_BITWIDTH: data_type.bitwidth})}
{generate_fifo_break_dv(f"{name}__elastic_fifo_inner", {ATTR_SLOTS: slots, ATTR_BITWIDTH: data_type.bitwidth})}
"""
