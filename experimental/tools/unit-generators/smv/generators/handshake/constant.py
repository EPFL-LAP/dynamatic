from generators.support.utils import *


def generate_constant(name, params):
  return _generate_constant(name, params["value"], mlir_type_to_smv_type(params["data_type"]))


def _generate_constant(name, value, data_type):
  return f"""
MODULE {name}(ins_valid, outs_ready)

  // output
  DEFINE ins_ready := {smv_format_constant(value, data_type)};
  DEFINE outs_valid := ins_valid;
  DEFINE outs :=
"""


if __name__ == "__main__":
  print(generate_constant("test_constant", {
        "value": 34, "data_type": "!handshake.channel<i32>"}))
