from generators.support.utils import *
from generators.support.tehb import generate_tehb


def generate_load(name, params):
    data_type = SmvScalarType(params[ATTR_PORT_TYPES]["dataOut"])
    addr_type = SmvScalarType(params[ATTR_PORT_TYPES]["addrIn"])

    return _generate_load(name, data_type, addr_type)


def _generate_load(name, data_type, addr_type):
    return f"""
MODULE {name}(addrIn, addrIn_valid, dataFromMem, dataFromMem_valid, addrOut_ready, dataOut_ready)
  VAR
  inner_addr_tehb : {name}__addr_tehb(addr_in, addr_in_valid, addr_out_ready);
  inner_data_tehb : {name}__data_tehb(data_from_mem, data_from_mem_valid, data_out_ready);

  //output
  DEFINE
  addrIn_ready := inner_addr_tehb.ins_ready;
  addrOut := inner_addr_tehb.outs;
  addrOut_valid := inner_addr_tehb.outs_valid;
  dataFromMem_ready := inner_data_tehb.ins_ready;
  dataOut := inner_data_tehb.outs;
  dataOut_valid := inner_data_tehb.outs_valid;

{generate_tehb(f"{name}__addr_tehb", {ATTR_DATA_TYPE: addr_type.mlir_type})}
{generate_tehb(f"{name}__data_tehb", {ATTR_DATA_TYPE: data_type.mlir_type})}
"""
