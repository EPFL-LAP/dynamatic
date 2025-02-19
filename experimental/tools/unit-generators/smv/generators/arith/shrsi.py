from generators.support.arith_utils import *
from generators.support.utils import *


def generate_shrsi(name, params):
  latency = params[ATTR_LATENCY]
  data_type = SmvScalarType(params[ATTR_DATA_TYPE])
  abstract_data = params[ATTR_ABSTRACT_DATA]

  if abstract_data:
    return generate_abstract_binary_op(name, latency, data_type)
  elif data_type.signed:
    return _generate_shrsi(name, latency, data_type)
  else:
    return _generate_shrsi_cast(name, latency, data_type)


def _generate_shrsi(name, latency, data_type):
  return f"""
{generate_binary_op_header(name)}
  DEFINE outs := lhs >> rhs;
  
  {generate_binary_op_handshake_manager(f"{name}__handshake_manager", {"latency": latency})}
"""


def _generate_shrsi_cast(name, latency, data_type):
  return f"""
{generate_binary_op_header(name)}
  DEFINE outs := signed(lhs) >> rhs;
  
  {generate_binary_op_handshake_manager(f"{name}__handshake_manager", {"latency": latency})}
"""
