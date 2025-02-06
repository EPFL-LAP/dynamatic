from generators.support.merge_notehb import (
    generate_merge_notehb,
)
from generators.handshake.buffer import generate_buffer
from generators.support.utils import SmvScalarType


def generate_merge(name, params):
  size = params["size"]
  data_type = SmvScalarType(params["data_type"])

  if data_type.bitwidth == 0:
    return _generate_merge_dataless(name, size)
  else:
    return _generate_merge(name, size, data_type)


def _generate_merge_dataless(name, size):
  return f"""
MODULE {name}({", ".join([f"ins_valid_{n}" for n in range(size)])}, outs_ready)
  VAR
  inner_tehb : {name}__tehb_dataless(inner_merge.outs_valid, outs_ready);
  inner_merge : {name}__merge_notehb_dataless({", ".join([f"ins_valid_{n}" for n in range(size)])}, inner_tehb.ins_ready);

  // output
  DEFINE
  {"\n  ".join([f"ins_ready_{n} := inner_merge.ins_ready{n};" for n in range(size)])}
  outs_valid = inner_tehb.outs_valid;

{generate_merge_notehb(f"{name}__merge_notehb_dataless", size)}
{generate_buffer(f"{name}__tehb_dataless", {"slots": 1, "timing": "R: 1", "data_type": "!handshake.control<>"})}
"""


def _generate_merge(name, size, data_type):
  return f"""
MODULE {name}({", ".join([f"ins_{n}, ins_valid{n}" for n in range(size)])}, outs_ready)
  VAR
  inner_tehb : {name}__tehb(inner_merge.outs, inner_merge.outs_valid, outs_ready);
  inner_merge : {name}__merge_notehb({", ".join([f"ins_{n}" for n in range(size)])}, {", ".join([f"ins_valid_{n}" for n in range(size)])}, inner_tehb.ins_ready);

  // output
  DEFINE
  {"\n  ".join([f"ins_ready_{n} := inner_merge.ins_ready{n};" for n in range(size)])}
  outs := inner_tehb.outs;
  outs_valid := inner_tehb.outs_valid;

{generate_merge_notehb(f"{name}__merge_notehb", size, data_type)}
{generate_buffer(f"{name}__tehb", {"slots": 1, "timing": "R: 1", "data_type": data_type.mlir_type})}
"""


if __name__ == "__main__":
  print(generate_merge("test_merge_dataless", {
        "size": 4, "data_type": "!handshake.control<>"}))
  print(generate_merge("test_merge", {
        "size": 2, "data_type": "!handshake.channel<i32>"}))
