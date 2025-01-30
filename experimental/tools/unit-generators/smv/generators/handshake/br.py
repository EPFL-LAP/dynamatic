from generators.support.utils import mlir_type_to_smv_type


def generate_br(name, params):
  if "data_type" not in params or params["data_type"] == "!handshake.control<>":
    return _generate_br_dataless(name)
  else:
    return _generate_br(name, mlir_type_to_smv_type(params["data_type"]))


def _generate_br_dataless(name):
  return f"""
MODULE {name}(ins_valid, outs_ready)

  // output
  DEFINE outs_valid :=  ins_valid;
  DEFINE ins_ready  :=  outs_ready;
"""


def _generate_br(name, data_type):
  return f"""
MODULE {name}(ins, ins_valid, outs_ready)

  // output
  DEFINE outs := ins;
  DEFINE outs_valid :=  ins_valid;
  DEFINE ins_ready  :=  outs_ready;
"""


if __name__ == "__main__":
  print(generate_br("test_br_dataless", {}))
  print(generate_br("test_br", {"data_type": "!handshake.channel<i32>"}))
