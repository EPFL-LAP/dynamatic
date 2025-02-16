from generators.support.utils import *


def generate_store(name, params):
  data_type = SmvScalarType(params[ATTR_DATA_TYPE])
  addr_type = SmvScalarType(params[ATTR_ADDR_TYPE])

  return _generate_store(name, data_type, addr_type)


def _generate_store(name, data_type, addr_type):
  return f"""
MODULE {name}(data_in, data_in_valid, addr_in, addr_in_valid, data_to_mem_ready, addr_out_ready)

  //output
  DEFINE
  data_in_ready := data_to_mem_ready;
  addr_in_ready := addr_out_ready;
  data_to_mem := data_in;
  data_to_mem_valid := data_in_valid;
  addr_out := addr_in;
  addr_out_valid := addr_in_valid;
"""


if __name__ == "__main__":
  print(generate_store("test_store", {
        "data_type": "!handshake.channel<i32>", "addr_type": "!handshake.channel<i32>"}))
