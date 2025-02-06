from generators.support.utils import SmvScalarType


def generate_br(name, params):
  data_type = SmvScalarType(params["data_type"])

  if data_type.bitwidth == 0:
    return _generate_br_dataless(name)
  else:
    return _generate_br(name, data_type)


def _generate_br_dataless(name):
  return f"""
MODULE {name}(ins_valid, outs_ready)

  // output
  DEFINE  
  outs_valid :=  ins_valid;
  ins_ready  :=  outs_ready;
"""


def _generate_br(name, data_type):
  return f"""
MODULE {name}(ins, ins_valid, outs_ready)

  // output
  DEFINE
  outs := ins;
  outs_valid :=  ins_valid;
  ins_ready  :=  outs_ready;
"""


if __name__ == "__main__":
  print(generate_br("test_br_dataless", {"data_type": "!handshake.control<>"}))
  print(generate_br("test_br", {"data_type": "!handshake.channel<i32>"}))
