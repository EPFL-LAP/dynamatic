from generators.support.utils import *
from generators.handshake.buffers.one_slot_break_r import generate_one_slot_break_r


def generate_load(name, params):
    data_type = SmvScalarType(params[ATTR_DATA_BITWIDTH])
    addr_type = SmvScalarType(params[ATTR_ADDR_BITWIDTH])

    return _generate_load(name, data_type, addr_type)


def _generate_load(name, data_type, addr_type):
    return f"""
MODULE {name}(addrIn, addrIn_valid, dataFromMem, dataFromMem_valid, addrOut_ready, dataOut_ready)
  VAR
  inner_addr_one_slot_break_r : {name}__addr_one_slot_break_r(addrIn, addrIn_valid, addrOut_ready);
  inner_data_one_slot_break_r : {name}__data_one_slot_break_r(dataFromMem, dataFromMem_valid, dataOut_ready);

  -- outputs
  DEFINE
  addrIn_ready := inner_addr_one_slot_break_r.ins_ready;
  addrOut := inner_addr_one_slot_break_r.outs;
  addrOut_valid := inner_addr_one_slot_break_r.outs_valid;
  dataFromMem_ready := inner_data_one_slot_break_r.ins_ready;
  dataOut := inner_data_one_slot_break_r.outs;
  dataOut_valid := inner_data_one_slot_break_r.outs_valid;

{generate_one_slot_break_r(f"{name}__addr_one_slot_break_r", {ATTR_BITWIDTH: addr_type.bitwidth})}
{generate_one_slot_break_r(f"{name}__data_one_slot_break_r", {ATTR_BITWIDTH: data_type.bitwidth})}
"""
