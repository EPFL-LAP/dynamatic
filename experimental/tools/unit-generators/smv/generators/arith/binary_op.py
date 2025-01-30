from generators.support.utils import mlir_type_to_smv_type
from generators.support.delay_buffer import generate_delay_buffer
from generators.handshake.join import generate_join

latency = {"addi": 0, "muli": 4}

symbol = {"addi": "+", "muli": "*"}


def generate_binary_op(name, params):
  if latency[params["op_type"]] == 0:
    return _generate_binary_op(name, params["op_type"], mlir_type_to_smv_type(params["data_type"]))
  else:
    return _generate_binary_op_lat(name, params["op_type"], mlir_type_to_smv_type(params["data_type"]), latency[params["op_type"]])


def _generate_binary_op(name, op_type, data_type):
  return f"""
MODULE {name}(lhs, lhs_valid, rhs, rhs_valid, result_ready)
  VAR inner_join : {name}__join(lhs_valid, rhs_valid, outs_ready);

  // output
  DEFINE lhs_ready := inner_join.ins_ready_0;
  DEFINE rhs_ready := inner_join.ins_ready_1;
  DEFINE outs_valid := inner_join.outs_valid;
  DEFINE outs := lhs {symbol[op_type]} rhs;
  
  {generate_join(f"{name}__join", {"size": 2})}
"""


def _generate_binary_op_lat(name, op_type, data_type, latency):
  return f"""
MODULE {name}(lhs, lhs_valid, rhs, rhs_valid, result_ready)
  VAR inner_join : {name}__join(lhs_valid, rhs_valid, outs_ready);
  VAR inner_delay_buffer : {name}__delay_buffer(inner_join.outs_valid, outs_ready);

  // output
  DEFINE lhs_ready := inner_join.ins_ready_0;
  DEFINE rhs_ready := inner_join.ins_ready_1;
  DEFINE outs_valid := inner_delay_buffer.outs_valid;
  DEFINE outs := lhs {symbol[op_type]} rhs;
  
  {generate_join(f"{name}__join", {"size": 2})}
  {generate_delay_buffer(f"{name}__delay_buffer", latency)}
"""


if __name__ == "__main__":
  print(generate_binary_op(
      "test_op", {"op_type": "addi", "data_type": "!handshake.channel<i32>"}))
  print(generate_binary_op("test_op_lat", {
        "op_type": "muli", "data_type": "!handshake.channel<i32>"}))
