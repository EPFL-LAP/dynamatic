from generators.support.arith_utils import generate_abstract_unary_op
from generators.support.utils import *


def generate_sitofp(name, params):
    latency = params[ATTR_LATENCY]
    output_type = SmvScalarType(params[ATTR_BITWIDTH])
    abstract_data = params[ATTR_ABSTRACT_DATA]

    if abstract_data:
        return generate_abstract_unary_op(name, latency, output_type)
    else:
        raise ValueError("Floating point operations support abstract data only")
