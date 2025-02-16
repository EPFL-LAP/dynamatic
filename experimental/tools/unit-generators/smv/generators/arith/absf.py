from generators.support.delay_buffer import generate_delay_buffer
from generators.support.utils import *


def generate_absf(name, params):
  latency = params[ATTR_LATENCY]
  data_type = SmvScalarType(params[ATTR_DATA_TYPE])

  return _generate_absf(name, latency, data_type)


def _generate_absf(name, latency, data_type):
  return f"""
MODULE {name}(ins, ins_valid, outs_ready)
  VAR inner_delay_buffer : {name}__delay_buffer(ins_valid, outs_ready);

  // output
  DEFINE ins_ready := inner_delay_buffer.ins_ready;
  DEFINE outs_valid := inner_delay_buffer.outs_valid;
  DEFINE outs := ins;
  
  {generate_delay_buffer(f"{name}__delay_buffer", latency)}
"""
