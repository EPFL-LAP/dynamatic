from generators.support.arith_utils import generate_abstract_binary_op
from generators.support.utils import *


def generate_maximumf(name, params):
    latency = params[ATTR_LATENCY]
    is_double = params[ATTR_IS_DOUBLE]
    abstract_data = params[ATTR_ABSTRACT_DATA]

    data_type = SmvScalarType(64 if is_double else 32)

    if abstract_data:
        return generate_abstract_binary_op(name, latency, data_type)
    else:
        raise ValueError("Floating point operations support abstract data only")
