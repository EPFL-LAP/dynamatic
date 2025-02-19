from generators.support.arith_utils import *
from generators.support.utils import *


def generate_shrui(name, params):
  latency = params[ATTR_LATENCY]
  data_type = SmvScalarType(params[ATTR_PORT_TYPES]["outs"])
  abstract_data = params[ATTR_ABSTRACT_DATA]

  if abstract_data:
    return generate_abstract_binary_op(name, latency, data_type)
  if data_type.signed:
    return _generate_shrui(name, latency, data_type)
  else:
    return _generate_shrui_cast(name, latency, data_type)


def _generate_shrui(name, latency, data_type):
  return f"""
{generate_binary_op_header(name)}
  DEFINE outs := lhs >> rhs;
  
  {generate_binary_op_handshake_manager(f"{name}__handshake_manager", {ATTR_LATENCY: latency})}
"""


def _generate_shrui_cast(name, latency, data_type):
  return f"""
{generate_binary_op_header(name)}
  DEFINE outs := unsigned(lhs) >> rhs;
  
  {generate_binary_op_handshake_manager(f"{name}__handshake_manager", {ATTR_LATENCY: latency})}
"""
