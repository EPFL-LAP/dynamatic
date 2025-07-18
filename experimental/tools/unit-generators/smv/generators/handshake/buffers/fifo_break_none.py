from generators.support.utils import *
import fifo_break_dv


def generate_fifo_break_none(name, params):
    slots = params[ATTR_SLOTS]
    data_type = SmvScalarType(params[ATTR_BITWIDTH])

    if data_type.bitwidth == 0:
        return _generate_fifo_break_none_dataless(name, slots)
    else:
        return _generate_fifo_break_none(name, slots, data_type)


def _generate_fifo_break_none_dataless(name, slots):
    return f"""
MODULE {name} (ins_valid, outs_ready)
  VAR
  inner_elastic_fifo : {name}__fifo_break_dv_dataless(fifo_valid, fifo_ready);

  DEFINE
  fifo_valid := ins_valid & (!outs_ready | inner_elastic_fifo.outs_valid);
  fifo_ready := outs_ready;

  -- output
  DEFINE
  ins_ready := inner_elastic_fifo.ins_ready | outs_ready;
  outs_valid := ins_valid | inner_elastic_fifo.outs_valid;

{fifo_break_dv(f"{name}__fifo_break_dv_dataless", {ATTR_SLOTS: slots, ATTR_BITWIDTH: 0})}
"""


def _generate_fifo_break_none(name, slots, data_type):
    return f"""
MODULE {name} (ins, ins_valid, outs_ready)
  VAR
  inner_elastic_fifo : {name}__elastic_fifo_inner(ins, fifo_valid, fifo_ready);

  DEFINE
  fifo_valid := ins_valid & (!outs_ready | inner_elastic_fifo.outs_valid);
  fifo_ready := outs_ready;

  -- output
  DEFINE
  ins_ready := inner_elastic_fifo.ins_ready | outs_ready;
  outs_valid := ins_valid | inner_elastic_fifo.outs_valid;
  outs := inner_elastic_fifo.outs_valid ? inner_elastic_fifo.outs : ins;

{fifo_break_dv(f"{name}__fifo_break_dv_inner", {ATTR_SLOTS: slots, ATTR_BITWIDTH: data_type.bitwidth})}
"""
