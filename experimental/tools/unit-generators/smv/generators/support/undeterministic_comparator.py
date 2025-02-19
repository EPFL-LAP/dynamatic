from generators.support.utils import *
from generators.support.arith_utils import *


def generate_undeterministic_comparator(name, params):
  latency = params[ATTR_LATENCY]
  data_type = SmvScalarType(params[ATTR_PORT_TYPES]["lhs"])

  return _generate_undeterministic_comparator(name, latency, data_type)


def _generate_undeterministic_comparator(name, latency, data_type):
  return f"""
MODULE {name}(lhs, lhs_valid, rhs, rhs_valid, outs_ready)
  VAR inner_handshake_manager : {name}__handshake_manager(lhs_valid, rhs_valid, outs_ready);
  VAR rand : boolean;

  ASSIGN
  init(rand) := {{TRUE, FALSE}};
  next(rand) := outs_valid & outs_ready ? {{TRUE, FALSE}} : rand;

  // output
  DEFINE lhs_ready := inner_handshake_manager.lhs_ready;
  DEFINE rhs_ready := inner_handshake_manager.rhs_ready;
  DEFINE outs_valid := inner_handshake_manager.outs_valid;
  DEFINE outs := rand;
  
  {generate_binary_op_handshake_manager(f"{name}__handshake_manager", {ATTR_LATENCY: latency})}
"""
