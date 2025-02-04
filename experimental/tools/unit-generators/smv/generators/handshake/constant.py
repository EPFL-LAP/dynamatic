from generators.support.utils import SmvScalarType


def generate_constant(name, params):
  value = params["value"]
  data_type = SmvScalarType(params["data_type"])

  return _generate_constant(name, value, data_type)


def _generate_constant(name, value, data_type):
  return f"""
MODULE {name}(ins_valid, outs_ready)

  // output
  DEFINE ins_ready := outs_ready;
  DEFINE outs_valid := ins_valid;
  DEFINE outs := {data_type.format_constant(value)};
"""


if __name__ == "__main__":
  print(generate_constant("test_constant", {
        "value": 34, "data_type": "!handshake.channel<i32>"}))
