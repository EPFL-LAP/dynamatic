from generators.support.delay_buffer import generate_delay_buffer
from generators.handshake.join import generate_join


def generate_binary_op_handshake_manager(name, latency):
  if latency == 0:
    return _generate_handshake_manager_no_lat(name)
  else:
    return _generate_handshake_manager(name, latency)


def _generate_handshake_manager_no_lat(name):
  return f"""
MODULE {name}(lhs_valid, rhs_valid, outs_ready)
  VAR inner_join : {name}__join(lhs_valid, rhs_valid, outs_ready);

  // output
  DEFINE lhs_ready := inner_join.ins_ready_0;
  DEFINE rhs_ready := inner_join.ins_ready_1;
  DEFINE outs_valid := inner_join.outs_valid;
  
  {generate_join(f"{name}__join", {"size": 2})}
"""


def _generate_handshake_manager(name, latency):
  return f"""
MODULE {name}(lhs_valid, rhs_valid, outs_ready)
  VAR inner_join : {name}__join(lhs_valid, rhs_valid, outs_ready);
  VAR inner_delay_buffer : {name}__delay_buffer(inner_join.outs_valid, outs_ready);

  // output
  DEFINE lhs_ready := inner_join.ins_ready_0;
  DEFINE rhs_ready := inner_join.ins_ready_1;
  DEFINE outs_valid := inner_delay_buffer.outs_valid;
  
  {generate_join(f"{name}__join", {"size": 2})}
  {generate_delay_buffer(f"{name}__delay_buffer", latency)}
"""
