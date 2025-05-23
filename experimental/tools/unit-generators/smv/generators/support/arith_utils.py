from generators.support.delay_buffer import generate_delay_buffer
from generators.handshake.join import generate_join
from generators.support.utils import *


def generate_binary_op_handshake_manager(name, params):
    latency = params[ATTR_LATENCY]

    if latency == 0:
        return _generate_handshake_manager_no_lat(name)
    else:
        return _generate_handshake_manager(name, latency)


def _generate_handshake_manager_no_lat(name):
    return f"""
MODULE {name}(lhs_valid, rhs_valid, outs_ready)
  VAR inner_join : {name}__join(lhs_valid, rhs_valid, outs_ready);

  -- output
  DEFINE lhs_ready := inner_join.ins_0_ready;
  DEFINE rhs_ready := inner_join.ins_1_ready;
  DEFINE outs_valid := inner_join.outs_valid;
  
  {generate_join(f"{name}__join", {"size": 2})}
"""


def _generate_handshake_manager(name, latency):
    return f"""
MODULE {name}(lhs_valid, rhs_valid, outs_ready)
  VAR inner_join : {name}__join(lhs_valid, rhs_valid, outs_ready);
  VAR inner_delay_buffer : {name}__delay_buffer(inner_join.outs_valid, outs_ready);

  -- output
  DEFINE lhs_ready := inner_join.ins_0_ready;
  DEFINE rhs_ready := inner_join.ins_1_ready;
  DEFINE outs_valid := inner_delay_buffer.outs_valid;
  
  {generate_join(f"{name}__join", {"size": 2})}
  {generate_delay_buffer(f"{name}__delay_buffer", {ATTR_LATENCY: latency})}
"""


def generate_binary_op_header(name):
    return f"""
MODULE {name}(lhs, lhs_valid, rhs, rhs_valid, outs_ready)
  VAR inner_handshake_manager : {name}__handshake_manager(lhs_valid, rhs_valid, outs_ready);

  -- output
  DEFINE lhs_ready := inner_handshake_manager.lhs_ready;
  DEFINE rhs_ready := inner_handshake_manager.rhs_ready;
  DEFINE result_valid := inner_handshake_manager.outs_valid;
"""


def generate_unanary_op_header(name):
    return f"""
MODULE {name}(ins, ins_valid, outs_ready)
  VAR inner_delay_buffer : {name}__delay_buffer(ins_valid, outs_ready);

  -- output
  DEFINE ins_ready := inner_delay_buffer.ins_ready;
  DEFINE outs_valid := inner_delay_buffer.outs_valid;
"""


def generate_abstract_binary_op(name, latency, data_type):
    return f"""
{generate_binary_op_header(name)}
  DEFINE result := {data_type.format_constant(0)};
  
  {generate_binary_op_handshake_manager(f"{name}__handshake_manager", {ATTR_LATENCY: latency})}
"""


def generate_abstract_unary_op(name, latency, data_type):
    return f"""
{generate_unanary_op_header(name)}
  DEFINE outs := {data_type.format_constant(0)};
  
  {generate_delay_buffer(f"{name}__delay_buffer", {ATTR_LATENCY: latency})}
"""
