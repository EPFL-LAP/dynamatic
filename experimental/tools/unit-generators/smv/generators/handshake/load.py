from generators.support.utils import *
from generators.handshake.buffer import generate_buffer


def generate_load(name, params):
  data_type = SmvScalarType(params[ATTR_DATA_TYPE])
  addr_type = SmvScalarType(params[ATTR_ADDR_TYPE])

  return _generate_load(name, data_type, addr_type)


def _generate_load(name, data_type, addr_type):
  return f"""
MODULE {name}(addr_in, addr_in_valid, data_from_mem, data_from_mem_valid, addr_out_ready, data_out_ready)
  VAR
  inner_addr_tehb : {name}__addr_tehb(addr_in, addr_in_valid, addr_out_ready);
  inner_data_tehb : {name}__data_tehb(data_from_mem, data_from_mem_valid, data_out_ready);

  //output
  DEFINE
  addr_in_ready := inner_addr_tehb.ins_ready;
  addr_out := inner_addr_tehb.outs;
  addr_out_valid := inner_addr_tehb.outs_valid;
  data_from_mem_ready := inner_data_tehb.ins_ready;
  data_out := inner_data_tehb.outs;
  data_out_valid := inner_data_tehb.outs_valid;

{generate_buffer(f"{name}__addr_tehb", {"slots": 1, "timing": "R: 1", "data_type": addr_type.mlir_type})}
{generate_buffer(f"{name}__data_tehb", {"slots": 1, "timing": "R: 1", "data_type": data_type.mlir_type})}
"""


if __name__ == "__main__":
  print(generate_load("test_load", {
        "data_type": "!handshake.channel<i32>", "addr_type": "!handshake.channel<i32>"}))
