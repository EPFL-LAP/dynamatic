from generators.support.arith_utils import generate_abstract_unary_op
from generators.support.utils import *


def generate_truncf(name, params):
  latency = params[ATTR_LATENCY]
  input_type = SmvScalarType(params[ATTR_PORT_TYPES]["ins"])
  output_type = SmvScalarType(params[ATTR_PORT_TYPES]["outs"])
  abstract_data = params[ATTR_ABSTRACT_DATA]

  if abstract_data:
    return generate_abstract_unary_op(name, latency, output_type)
  else:
    raise ValueError("Floating point operations support abstract data only")
