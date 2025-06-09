from generators.support.utils import *
from generators.support.arith_utils import *


def generate_nondeterministic_comparator(name, params):
    latency = params[ATTR_LATENCY]
    data_type = SmvScalarType(params[ATTR_BITWIDTH])

    return _generate_nondeterministic_comparator(name, latency, data_type)


def _generate_nondeterministic_comparator(name, latency, data_type):
    return f"""
MODULE {name}(lhs, lhs_valid, rhs, rhs_valid, result_ready)
  VAR inner_handshake_manager : {name}__handshake_manager(lhs_valid, rhs_valid, result_ready);
  VAR rand : boolean;

  ASSIGN
  init(rand) := {{TRUE, FALSE}};
  next(rand) := result_valid & result_ready ? {{TRUE, FALSE}} : rand;

  -- output
  DEFINE lhs_ready := inner_handshake_manager.lhs_ready;
  DEFINE rhs_ready := inner_handshake_manager.rhs_ready;
  DEFINE result_valid := inner_handshake_manager.outs_valid;
  DEFINE result := rand;
  
  {generate_binary_op_handshake_manager(f"{name}__handshake_manager", {ATTR_LATENCY: latency})}
"""
