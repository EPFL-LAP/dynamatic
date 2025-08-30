from generators.support.utils import *
from generators.handshake.buffers.fifo_break_none import generate_fifo_break_none
from generators.handshake.buffers.one_slot_break_dv import generate_one_slot_break_dv
from generators.support.buffer_counter import generate_buffer_counter


def generate_fifo_break_dv(name, params):
    slots = params[ATTR_SLOTS] if ATTR_SLOTS in params else 1
    data_type = SmvScalarType(params[ATTR_BITWIDTH])
    debug_counter = params.get(ATTR_DEBUG_COUNTER, False)

    if slots == 1:
        return generate_one_slot_break_dv(name, {ATTR_BITWIDTH: params[ATTR_BITWIDTH], ATTR_DEBUG_COUNTER: debug_counter})

    if data_type.bitwidth == 0:
        return _generate_fifo_break_dv_dataless(name, slots, debug_counter)
    else:
        return _generate_fifo_break_dv(name, slots, data_type, debug_counter)


def _generate_fifo_break_dv_dataless(name, slots, debug_counter):
    fifo_name = f"{name}__fifo"
    one_slot_name = f"{name}__break_dv"
    return f"""
MODULE {name}(ins_valid, outs_ready)
  VAR
    fifo : {fifo_name}(ins_valid, break_dv_ready);
    break_dv   : {one_slot_name}(fifo_valid, outs_ready);
    {f"debug_counter : {name}__debug_counter(ins_valid, ins_ready, outs_valid, outs_ready);" if debug_counter else ""}

  DEFINE
    fifo_valid := fifo.outs_valid;
    break_dv_ready := break_dv.ins_ready;

    ins_ready := fifo.ins_ready;
    outs_valid := break_dv.outs_valid;
    
{generate_fifo_break_none(fifo_name, {ATTR_SLOTS: slots - 1, ATTR_BITWIDTH: 0})}
{generate_one_slot_break_dv(one_slot_name, {ATTR_BITWIDTH: 0})}
{generate_buffer_counter(f"{name}__debug_counter", slots) if debug_counter else ""}
"""


def _generate_fifo_break_dv(name, slots, data_type, debug_counter):
    fifo_name = f"{name}__fifo"
    one_slot_name = f"{name}__break_dv"
    return f"""
MODULE {name}(ins, ins_valid, outs_ready)
  VAR
    fifo : {fifo_name}(ins, ins_valid, break_dv_ready);
    break_dv : {one_slot_name}(fifo_data, fifo_valid, outs_ready);
    {f"debug_counter : {name}__debug_counter(ins_valid, ins_ready, outs_valid, outs_ready);" if debug_counter else ""}

  DEFINE
    fifo_data := fifo.outs;
    fifo_valid := fifo.outs_valid;
    break_dv_ready := break_dv.ins_ready;

    ins_ready := fifo.ins_ready;
    outs_valid := break_dv.outs_valid;
    outs := break_dv.outs;

{generate_fifo_break_none(fifo_name, {ATTR_SLOTS: slots - 1, ATTR_BITWIDTH: data_type.bitwidth})}
{generate_one_slot_break_dv(one_slot_name, {ATTR_BITWIDTH: data_type.bitwidth})}
{generate_buffer_counter(f"{name}__debug_counter", slots) if debug_counter else ""}
"""
