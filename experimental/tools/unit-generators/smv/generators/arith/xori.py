from generators.support.binary_op_handshake_manager import generate_binary_op_handshake_manager
from generators.support.arith_op_headers import generate_binary_op_header
from generators.support.utils import *


def generate_xori(name, params):
  latency = params[ATTR_LATENCY]
  data_type = SmvScalarType(params[ATTR_DATA_TYPE])

  return _generate_xori(name, latency, data_type)


def _generate_xori(name, latency, data_type):
  return f"""
{generate_binary_op_header(name)}
  DEFINE outs := lhs xor rhs;
  
  {generate_binary_op_handshake_manager(f"{name}__handshake_manager", latency)}
"""
