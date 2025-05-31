from generators.support.utils import *
from generators.support.tehb import generate_tehb


def generate_load(name, params):
    data_type = SmvScalarType(params[ATTR_DATA_BITWIDTH])
    addr_type = SmvScalarType(params[ATTR_ADDR_BITWIDTH])

    return _generate_load(name, data_type, addr_type)


def _generate_load(name, data_type, addr_type):
    return f"""
MODULE {name}(addrIn, addrIn_valid, dataFromMem, dataFromMem_valid, addrOut_ready, dataOut_ready)
  VAR
  inner_addr_tehb : {name}__addr_tehb(addrIn, addrIn_valid, addrOut_ready);
  inner_data_tehb : {name}__data_tehb(dataFromMem, dataFromMem_valid, dataOut_ready);

  //output
  DEFINE
  addrIn_ready := inner_addr_tehb.ins_ready;
  addrOut := inner_addr_tehb.outs;
  addrOut_valid := inner_addr_tehb.outs_valid;
  dataFromMem_ready := inner_data_tehb.ins_ready;
  dataOut := inner_data_tehb.outs;
  dataOut_valid := inner_data_tehb.outs_valid;

{generate_tehb(f"{name}__addr_tehb", {ATTR_BITWIDTH: addr_type.bitwidth})}
{generate_tehb(f"{name}__data_tehb", {ATTR_BITWIDTH: data_type.bitwidth})}
"""
