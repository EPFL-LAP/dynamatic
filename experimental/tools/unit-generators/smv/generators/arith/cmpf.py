from generators.support.arith_utils import *
from generators.support.utils import *


def generate_cmpf(name, params):
  predicate = params[ATTR_PREDICATE]
  symbol = get_symbol_from_predicate(predicate)
  latency = params[ATTR_LATENCY]
  data_type = SmvScalarType(params[ATTR_DATA_TYPE])

  return _generate_cmpf(name, latency, symbol, data_type)


def _generate_cmpf(name, latency, symbol, data_type):
  return f"""
{generate_binary_op_header(name)}
  DEFINE outs := {{TRUE, FALSE}};
  
  {generate_binary_op_handshake_manager(f"{name}__handshake_manager", {"latency": latency})}
"""


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
