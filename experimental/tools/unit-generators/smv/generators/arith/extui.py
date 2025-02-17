from generators.support.delay_buffer import generate_delay_buffer
from generators.support.arith_op_headers import generate_unanary_op_header
from generators.support.utils import *


def generate_extui(name, params):
  latency = params[ATTR_LATENCY]
  input_type = SmvScalarType(params[ATTR_INPUT_TYPE])
  output_type = SmvScalarType(params[ATTR_OUTPUT_TYPE])

  return _generate_extui(name, latency, input_type, output_type)


def _generate_extui(name, latency, input_type, output_type):
  return f"""
{generate_unanary_op_header(name)}
  DEFINE outs := extend(ins, {output_type.bitwidth - input_type.bitwidth});
  
  {generate_delay_buffer(f"{name}__delay_buffer", {"latency": latency})}
"""
