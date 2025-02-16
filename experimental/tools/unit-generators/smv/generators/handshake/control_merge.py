from generators.support.merge_notehb import (
    generate_merge_notehb,
)
from generators.handshake.buffer import generate_buffer
from generators.handshake.fork import generate_fork
from generators.support.utils import *


def generate_control_merge(name, params):
  size = params["size"]
  data_type = SmvScalarType(params["data_type"])
  index_type = SmvScalarType(params["index_type"])

  if data_type.bitwidth == 0:
    return _generate_control_merge_dataless(name, size, index_type)
  else:
    return _generate_control_merge(name, size, index_type, data_type)


def _generate_control_merge_dataless(name, size, index_type):
  return f"""
MODULE {name}({", ".join([f"ins_valid_{n}" for n in range(size)])}, outs_ready, index_ready)
  VAR
  inner_tehb : {name}__tehb(index_in, inner_merge.outs_valid, inner_fork.ins_ready);
  inner_merge : {name}__merge_notehb_dataless({", ".join([f"ins_valid_{n}" for n in range(size)])}, inner_tehb.ins_ready);
  inner_fork : {name}__fork_dataless(inner_tehb.outs_valid, outs_ready, index_ready);

  DEFINE
  index_in := case
    {"\n    ".join([f"ins_valid_{n} = TRUE : {index_type.format_constant(n)};" for n in range(size)])}
    TRUE: {index_type.format_constant(0)};
  esac;

  // output
  DEFINE
  {"\n  ".join([f"ins_ready_{n} := inner_merge.ins_ready_{n};" for n in range(size)])}
  outs_valid := inner_fork.outs_valid_0;
  index_valid := inner_fork.outs_valid_1;
  index := inner_tehb.outs;

{generate_merge_notehb(f"{name}__merge_notehb_dataless", size)}
{generate_buffer(f"{name}__tehb", {"slots": 1, "timing": "R: 1", "data_type": index_type.mlir_type})}
{generate_fork(f"{name}__fork_dataless", {"size": 2, "data_type": HANSHAKE_CONTROL_TYPE})}
"""


def _generate_control_merge(name, size, index_type, data_type):
  return f"""
MODULE {name}({", ".join([f"ins_{n}, ins_valid_{n}" for n in range(size)])}, outs_ready, index_ready)
  VAR
  inner_control_merge : {name}__control_merge_dataless({", ".join([f"ins_valid_{n}" for n in range(size)])}, outs_ready, index_ready);

  DEFINE
  data := case
    {"\n    ".join([f"ins_valid_{n} : ins_{n};" for n in range(size)])}
    TRUE: {data_type.format_constant(0)};
  esac;

  // output
  DEFINE
  {"\n  ".join([f"ins_ready_{n} := inner_control_merge.ins_ready_{n};" for n in range(size)])}
  outs_valid := inner_control_merge.outs_valid;
  index_valid := inner_control_merge.index_valid;
  outs := data;
  index := inner_control_merge.index;

{_generate_control_merge_dataless(f"{name}__control_merge_dataless", size, index_type)}
"""


if __name__ == "__main__":
  print(generate_control_merge("test_control_merge_dataless",
        {"size": 4, "index_type": "!handshake.channel<i32>", "data_type": "!handshake.control<>"}))
  print(generate_control_merge(
      "test_control_merge_fork", {"size": 2, "index_type": "!handshake.channel<i1>", "data_type": "!handshake.channel<i32>"}))
