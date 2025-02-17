from generators.support.delay_buffer import generate_delay_buffer
from generators.support.arith_op_headers import generate_unanary_op_header
from generators.support.utils import *


def generate_trunci(name, params):
  latency = params[ATTR_LATENCY]
  input_type = SmvScalarType(params[ATTR_INPUT_TYPE])
  output_type = SmvScalarType(params[ATTR_OUTPUT_TYPE])

  return _generate_trunci(name, latency, input_type, output_type)


def _generate_trunci(name, latency, input_type, output_type):
  return f"""
{generate_unanary_op_header(name)}
  DEFINE outs := ins[{output_type.bitwidth}:0];
  
  {generate_delay_buffer(f"{name}__delay_buffer", {"latency": latency})}
"""
