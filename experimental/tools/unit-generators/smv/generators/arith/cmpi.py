from generators.support.binary_op_handshake_manager import generate_binary_op_handshake_manager
from generators.support.utils import *


def generate_cmpi(name, params):
  predicate = params[ATTR_PREDICATE]
  symbol = get_symbol_from_predicate(predicate)
  sign = get_sign_from_predicate(predicate)
  latency = params[ATTR_LATENCY]
  data_type = SmvScalarType(params[ATTR_DATA_TYPE])

  if sign is None or data_type.smv_type.split()[0] == sign:
    return _generate_cmpi(name, latency, symbol, data_type)
  else:
    modifier = sign
    return _generate_cmpi_cast(name, latency, symbol, modifier, data_type)


def _generate_cmpi(name, latency, symbol, data_type):
  return f"""
MODULE {name}(lhs, lhs_valid, rhs, rhs_valid, outs_ready)
  VAR inner_handshake_manager : {name}__handshake_manager(lhs_valid, rhs_valid, outs_ready);

  // output
  DEFINE lhs_ready := inner_handshake_manager.lhs_ready;
  DEFINE rhs_ready := inner_handshake_manager.rhs_ready;
  DEFINE outs_valid := inner_handshake_manager.outs_valid;
  DEFINE outs := lhs {symbol} rhs;
  
  {generate_binary_op_handshake_manager(f"{name}__handshake_manager", latency)}
"""


def _generate_cmpi_cast(name, latency, symbol, modifier, data_type):
  return f"""
MODULE {name}(lhs, lhs_valid, rhs, rhs_valid, outs_ready)
  VAR inner_handshake_manager : {name}__handshake_manager(lhs_valid, rhs_valid, outs_ready);

  // output
  DEFINE lhs_ready := inner_handshake_manager.lhs_ready;
  DEFINE rhs_ready := inner_handshake_manager.rhs_ready;
  DEFINE outs_valid := inner_handshake_manager.outs_valid;
  DEFINE outs := ({modifier})lhs {symbol} ({modifier})rhs;
  
  {generate_binary_op_handshake_manager(f"{name}__handshake_manager", latency)}
"""


def get_symbol_from_predicate(pred):
  match pred:
    case "eq":
      return "="
    case "neq":
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
    case "eq" | "neq":
      return None
    case "slt" | "sle" | "sgt" | "sge":
      return "signed"
    case "ult" | "ule" | "ugt" | "uge":
      return "unsigned"
    case _:
      raise ValueError(f"Predicate {pred} not known")
