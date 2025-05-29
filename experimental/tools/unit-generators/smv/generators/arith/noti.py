from generators.support.arith_utils import *
from generators.support.utils import *


def generate_not(name, params):
    latency = params[ATTR_LATENCY]
    data_type = SmvScalarType(params[ATTR_BITWIDTH])
    abstract_data = params[ATTR_ABSTRACT_DATA]

    if abstract_data:
        return generate_abstract_unary_op(name, latency, data_type)
    else:
        return _generate_not(name, latency, data_type)


def _generate_not(name, latency, data_type):
    return f"""
{generate_unanary_op_header(name)}
  DEFINE outs := !ins;
  
  {generate_delay_buffer(f"{name}__delay_buffer", {ATTR_LATENCY: latency})}
"""
