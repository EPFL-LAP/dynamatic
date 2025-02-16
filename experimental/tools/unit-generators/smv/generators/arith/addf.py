from generators.support.binary_op_handshake_manager import generate_binary_op_handshake_manager
from generators.support.utils import *


def generate_addf(name, params):
  latency = params[ATTR_LATENCY]
  data_type = SmvScalarType(params[ATTR_DATA_TYPE])

  return _generate_addf(name, latency, data_type)


def _generate_addf(name, latency, data_type):
  return f"""
MODULE {name}(lhs, lhs_valid, rhs, rhs_valid, outs_ready)
  VAR inner_handshake_manager : {name}__handshake_manager(lhs_valid, rhs_valid, outs_ready);

  // output
  DEFINE lhs_ready := inner_handshake_manager.lhs_ready;
  DEFINE rhs_ready := inner_handshake_manager.rhs_ready;
  DEFINE outs_valid := inner_handshake_manager.outs_valid;
  DEFINE outs := lhs;
  
  {generate_binary_op_handshake_manager(f"{name}__handshake_manager", latency)}
"""
