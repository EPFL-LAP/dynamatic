from generators.support.arith_utils import generate_abstract_binary_op
from generators.support.utils import *


def generate_divf(name, params):
  latency = params[ATTR_LATENCY]
  data_type = SmvScalarType(params[ATTR_PORT_TYPES]["outs"])
  abstract_data = params[ATTR_ABSTRACT_DATA]

  if abstract_data:
    return generate_abstract_binary_op(name, latency, data_type)
  else:
    raise ValueError("Floating point operations support abstract data only")
