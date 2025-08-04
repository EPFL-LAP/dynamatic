from generators.support.utils import *
from generators.handshake.buffers.fifo_break_none import generate_fifo_break_none
from generators.handshake.buffers.one_slot_break_dv import generate_one_slot_break_dv


def generate_fifo_break_dv(name, params):
    slots = params[ATTR_SLOTS] if ATTR_SLOTS in params else 1
    data_type = SmvScalarType(params[ATTR_BITWIDTH])

    if slots == 1:
        return generate_one_slot_break_dv(name, {ATTR_BITWIDTH: params[ATTR_BITWIDTH]})

    if data_type.bitwidth == 0:
        return _generate_fifo_break_dv_dataless(name, slots)
    else:
        return _generate_fifo_break_dv(name, slots, data_type)


def _generate_fifo_break_dv_dataless(name, slots):
    fifo_name = f"{name}__fifo"
    one_slot_name = f"{name}__break_dv"
    return f"""
MODULE {name}(ins_valid, outs_ready)
  VAR
    fifo : {fifo_name}(ins_valid, break_dv_ready);
    break_dv   : {one_slot_name}(fifo_valid, outs_ready);

  DEFINE
    fifo_valid := fifo.outs_valid;
    break_dv_ready := break_dv.ins_ready;

    ins_ready := fifo.ins_ready;
    outs_valid := break_dv.outs_valid;
    
{generate_fifo_break_none(fifo_name, {ATTR_SLOTS: slots - 1, ATTR_BITWIDTH: 0})}
{generate_one_slot_break_dv(one_slot_name, {ATTR_BITWIDTH: 0})}
"""


def _generate_fifo_break_dv(name, slots, data_type):
    fifo_name = f"{name}__fifo"
    one_slot_name = f"{name}__break_dv"
    return f"""
MODULE {name}(ins, ins_valid, outs_ready)
  VAR
    fifo : {fifo_name}(ins, ins_valid, break_dv_ready);
    break_dv : {one_slot_name}(fifo_data, fifo_valid, outs_ready);

  DEFINE
    fifo_data := fifo.outs;
    fifo_valid := fifo.outs_valid;
    break_dv_ready := break_dv.ins_ready;

    ins_ready := fifo.ins_ready;
    outs_valid := break_dv.outs_valid;
    outs := break_dv.outs;

{generate_fifo_break_none(fifo_name, {ATTR_SLOTS: slots - 1, ATTR_BITWIDTH: data_type.bitwidth})}
{generate_one_slot_break_dv(one_slot_name, {ATTR_BITWIDTH: data_type.bitwidth})}
"""
