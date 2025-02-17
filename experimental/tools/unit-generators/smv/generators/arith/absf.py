from generators.support.delay_buffer import generate_delay_buffer
from generators.support.arith_op_headers import generate_unanary_op_header
from generators.support.utils import *


def generate_absf(name, params):
  latency = params[ATTR_LATENCY]
  data_type = SmvScalarType(params[ATTR_DATA_TYPE])

  return _generate_absf(name, latency, data_type)


def _generate_absf(name, latency, data_type):
  return f"""
{generate_unanary_op_header(name)}
  DEFINE outs := ins;
  
  {generate_delay_buffer(f"{name}__delay_buffer", {"latency": latency})}
"""
