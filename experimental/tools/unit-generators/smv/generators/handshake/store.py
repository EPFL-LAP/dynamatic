from generators.support.utils import *


def generate_store(name, params):
  data_type = SmvScalarType(params[ATTR_PORT_TYPES]["data_in"])
  addr_type = SmvScalarType(params[ATTR_PORT_TYPES]["addr_in"])

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
