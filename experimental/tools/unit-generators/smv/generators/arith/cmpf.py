from generators.support.nondeterministic_comparator import generate_nondeterministic_comparator
from generators.support.utils import *


def generate_cmpf(name, params):
    predicate = params[ATTR_PREDICATE]
    symbol = get_symbol_from_predicate(predicate)
    latency = params[ATTR_LATENCY]
    is_double = params[ATTR_IS_DOUBLE]
    abstract_data = params[ATTR_ABSTRACT_DATA]
    params[ATTR_BITWIDTH] = 64 if is_double else 32

    if abstract_data:
        return generate_nondeterministic_comparator(name, params)
    else:
        raise ValueError(
            "Floating point operations support abstract data only")


def get_symbol_from_predicate(pred):
    match pred:
        case "oeq" | "ueq":
            return "="
        case "one" | "une":
            return "!="
        case "olt" | "ult":
            return "<"
        case "ole" | "ule":
            return "<="
        case "ogt" | "ugt":
            return ">"
        case "oge" | "uge":
            return ">="
        case "uno":
            return None
        case _:
            raise ValueError(f"Predicate {pred} not known")
