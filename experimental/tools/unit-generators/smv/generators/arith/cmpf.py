from generators.support.binary_op_handshake_manager import generate_binary_op_handshake_manager
from generators.support.utils import SmvScalarType


def generate_cmpf(name, params):
  predicate = params["predicate"]
  symbol = get_symbol_from_predicate(predicate)
  latency = params["latency"]
  data_type = SmvScalarType(params["data_type"])

  return _generate_cmpf(name, latency, symbol, data_type)



def _generate_cmpf(name, latency, symbol, data_type):
  return f"""
MODULE {name}(lhs, lhs_valid, rhs, rhs_valid, outs_ready)
  VAR inner_handshake_manager : {name}__handshake_manager(lhs_valid, rhs_valid, outs_ready);
  FROZENVAR undetermined : {{TRUE, FALSE}};

  // output
  DEFINE lhs_ready := inner_handshake_manager.lhs_ready;
  DEFINE rhs_ready := inner_handshake_manager.rhs_ready;
  DEFINE outs_valid := inner_handshake_manager.outs_valid;
  DEFINE outs := undetermined;
  
  {generate_binary_op_handshake_manager(f"{name}__handshake_manager", latency)}
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
