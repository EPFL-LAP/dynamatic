from generators.support.utils import *
from generators.handshake.buffer import generate_buffer


def generate_load(name, params):
  return _generate_load(name, params["data_type"], params["addr_type"])


def _generate_load(name, data_type, addr_type):
  return f"""
MODULE {name}(addr_in, addr_in_valid, addr_out_ready, data_from_mem, data_from_mem_valid, data_out_ready)
  VAR inner_addr_tehb : {name}__addr_tehb(addr_in, addr_in_valid, addr_out_ready);
  VAR inner_data_tehb : {name}__data_tehb(data_from_mem, data_from_mem_valid, data_out_ready);

  //output
  DEFINE addr_in_ready := inner_addr_tehb.ins_ready;
  DEFINE addr_out := inner_addr_tehb.outs;
  DEFINE addr_out_valid := inner_addr_tehb.outs_valid;
  DEFINE data_from_mem_ready := inner_data_tehb.ins_ready;
  DEFINE data_out := inner_data_tehb.outs;
  DEFINE data_out_valid := inner_data_tehb.outs_valid;

{generate_buffer(f"{name}__addr_tehb", {"slots": 1, "timing": "R: 1", "data_type": addr_type})}
{generate_buffer(f"{name}__data_tehb", {"slots": 1, "timing": "R: 1", "data_type": data_type})}
"""


if __name__ == "__main__":
  print(generate_load("test_load", {
        "data_type": "!handshake.channel<i32>", "addr_type": "!handshake.channel<i32>"}))
