from generators.support.utils import mlir_type_to_smv_type


def generate_sink(name, params):
  if "data_type" not in params or params["data_type"] == "!handshake.control<>":
    return _generate_sink_dataless(name)
  else:
    return _generate_sink(name, mlir_type_to_smv_type(params["data_type"]))


def _generate_sink_dataless(name):
  return f"""
MODULE {name}(ins_valid)

  // output
  DEFINE ins_ready  :=  TRUE;
"""


def _generate_sink(name, data_type):
  return f"""
MODULE {name}(ins, ins_valid)

  // output
  DEFINE ins_ready  :=  TRUE;
"""


if __name__ == "__main__":
  print(generate_sink("test_sink_dataless", {}))
  print(generate_sink("test_sink", {"data_type": "!handshake.channel<i32>"}))
