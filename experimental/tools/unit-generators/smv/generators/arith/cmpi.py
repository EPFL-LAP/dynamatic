from generators.support.arith_utils import *
from generators.support.utils import *
from generators.support.nondeterministic_comparator import generate_nondeterministic_comparator


def generate_cmpi(name, params):
    predicate = params[ATTR_PREDICATE]
    symbol = get_symbol_from_predicate(predicate)
    sign = get_sign_from_predicate(predicate)
    latency = params[ATTR_LATENCY]
    data_type = SmvScalarType(params[ATTR_BITWIDTH])
    abstract_data = params[ATTR_ABSTRACT_DATA]

    if abstract_data:
        return generate_nondeterministic_comparator(name, params)
    elif sign is None or data_type.smv_type.split()[0] == sign:
        return _generate_cmpi(name, latency, symbol, data_type)
    else:
        modifier = sign
        return _generate_cmpi_cast(name, latency, symbol, modifier, data_type)


def _generate_cmpi(name, latency, symbol, data_type):
    return f"""
{generate_binary_op_header(name)}
  DEFINE result := lhs {symbol} rhs;
  
  {generate_binary_op_handshake_manager(f"{name}__handshake_manager", {ATTR_LATENCY: latency})}
"""


def _generate_cmpi_cast(name, latency, symbol, modifier, data_type):
    return f"""
{generate_binary_op_header(name)}
  DEFINE result := ({modifier})lhs {symbol} ({modifier})rhs;
  
  {generate_binary_op_handshake_manager(f"{name}__handshake_manager", {ATTR_LATENCY: latency})}
"""


def get_symbol_from_predicate(pred):
    match pred:
        case "eq":
            return "="
        case "ne":
            return "!="
        case "slt" | "ult":
            return "<"
        case "sle" | "ule":
            return "<="
        case "sgt" | "ugt":
            return ">"
        case "sge" | "uge":
            return ">="
        case _:
            raise ValueError(f"Predicate {pred} not known")


def get_sign_from_predicate(pred):
    match pred:
        case "eq" | "ne":
            return None
        case "slt" | "sle" | "sgt" | "sge":
            return "signed"
        case "ult" | "ule" | "ugt" | "uge":
            return "unsigned"
        case _:
            raise ValueError(f"Predicate {pred} not known")
