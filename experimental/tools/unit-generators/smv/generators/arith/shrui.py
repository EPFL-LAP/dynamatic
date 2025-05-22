from generators.support.arith_utils import *
from generators.support.utils import *


def generate_shrui(name, params):
  latency = params[ATTR_LATENCY]
  data_type = SmvScalarType(params[ATTR_BITWIDTH])
  abstract_data = params[ATTR_ABSTRACT_DATA]

  if abstract_data:
    return generate_abstract_binary_op(name, latency, data_type)
  else:
    return _generate_shrui(name, latency, data_type)


def _generate_shrui(name, latency, data_type):
  return f"""
{generate_binary_op_header(name)}
  DEFINE result := lhs >> rhs;
  
  {generate_binary_op_handshake_manager(f"{name}__handshake_manager", {ATTR_LATENCY: latency})}
"""


def _generate_shrui_cast(name, latency, data_type):
  return f"""
{generate_binary_op_header(name)}
  DEFINE result := unsigned(lhs) >> rhs;
  
  {generate_binary_op_handshake_manager(f"{name}__handshake_manager", {ATTR_LATENCY: latency})}
"""
