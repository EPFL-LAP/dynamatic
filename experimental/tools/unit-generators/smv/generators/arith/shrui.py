from generators.support.binary_op_handshake_manager import generate_binary_op_handshake_manager
from generators.support.arith_op_headers import generate_binary_op_header
from generators.support.utils import *


def generate_shrui(name, params):
  latency = params[ATTR_LATENCY]
  data_type = SmvScalarType(params[ATTR_DATA_TYPE])

  if not data_type.signed:
    return _generate_shrui(name, latency, data_type)
  else:
    return _generate_shrui_cast(name, latency, data_type)


def _generate_shrui(name, latency, data_type):
  return f"""
{generate_binary_op_header(name)}
  DEFINE outs := lhs >> rhs;
  
  {generate_binary_op_handshake_manager(f"{name}__handshake_manager", {"latency": latency})}
"""


def _generate_shrui_cast(name, latency, data_type):
  return f"""
{generate_binary_op_header(name)}
  DEFINE outs := unsigned(lhs) >> rhs;
  
  {generate_binary_op_handshake_manager(f"{name}__handshake_manager", {"latency": latency})}
"""
