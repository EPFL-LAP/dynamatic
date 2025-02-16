from generators.support.delay_buffer import generate_delay_buffer
from generators.support.utils import *


def generate_sitofp(name, params):
  latency = params[ATTR_LATENCY]
  input_type = SmvScalarType(params[ATTR_INPUT_TYPE])
  output_type = SmvScalarType(params[ATTR_OUTPUT_TYPE])

  return _generate_sitofp(name, latency, input_type, output_type)


def _generate_sitofp(name, latency, input_type, output_type):
  return f"""
MODULE {name}(ins, ins_valid, outs_ready)
  VAR inner_delay_buffer : {name}__delay_buffer(ins_valid, outs_ready);

  // output
  DEFINE ins_ready := inner_delay_buffer.ins_ready;
  DEFINE outs_valid := inner_delay_buffer.outs_valid;
  DEFINE outs := ins;
  
  {generate_delay_buffer(f"{name}__delay_buffer", latency)}
"""
