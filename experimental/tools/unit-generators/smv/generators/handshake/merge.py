from generators.support.merge_notehb import (
    generate_merge_notehb,
)
from generators.support.tehb import generate_tehb
from generators.support.utils import *


def generate_merge(name, params):
  size = params[ATTR_SIZE]
  data_type = SmvScalarType(params[ATTR_PORT_TYPES]["outs"])

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
  {"\n  ".join([f"ins_ready_{n} := inner_merge.ins_ready_{n};" for n in range(size)])}
  outs_valid := inner_tehb.outs_valid;

{generate_merge_notehb(f"{name}__merge_notehb_dataless", {ATTR_SIZE: size, ATTR_DATA_TYPE: HANDSHAKE_CONTROL_TYPE.mlir_type})}
{generate_tehb(f"{name}__tehb_dataless", {ATTR_DATA_TYPE: HANDSHAKE_CONTROL_TYPE.mlir_type})}
"""


def _generate_merge(name, size, data_type):
  return f"""
MODULE {name}({", ".join([f"ins_{n}, ins_valid_{n}" for n in range(size)])}, outs_ready)
  VAR
  inner_tehb : {name}__tehb(inner_merge.outs, inner_merge.outs_valid, outs_ready);
  inner_merge : {name}__merge_notehb({", ".join([f"ins_{n}" for n in range(size)])}, {", ".join([f"ins_valid_{n}" for n in range(size)])}, inner_tehb.ins_ready);

  // output
  DEFINE
  {"\n  ".join([f"ins_ready_{n} := inner_merge.ins_ready_{n};" for n in range(size)])}
  outs := inner_tehb.outs;
  outs_valid := inner_tehb.outs_valid;

{generate_merge_notehb(f"{name}__merge_notehb", {ATTR_SIZE: size, ATTR_DATA_TYPE: data_type.mlir_type})}
{generate_tehb(f"{name}__tehb_dataless", {ATTR_DATA_TYPE: data_type.mlir_type})}
"""
