from generators.handshake.buffer import generate_buffer
from generators.support.utils import *


def generate_mux(name, params):
  size = params[ATTR_SIZE]
  data_type = SmvScalarType(params[ATTR_DATA_TYPE])
  select_type = SmvScalarType(params[ATTR_SELECT_TYPE])

  if data_type.bitwidth == 0:
    return _generate_mux_dataless(name, size, select_type)
  else:
    return _generate_mux(name, size, data_type, select_type)


def _generate_mux_dataless(name, size, select_type):
  return f"""
MODULE {name}({", ".join([f"ins_valid_{n}" for n in range(size)])}, index, index_valid, outs_ready)
  VAR
  inner_tehb : {name}__tehb_dataless(tehb_ins_valid, outs_ready);

  DEFINE
  tehb_ins_valid := case
    {"\n    ".join([f"index = {select_type.format_constant(n)} : index_valid & ins_valid_{n};" for n in range(size)])}
    TRUE : FALSE;
  esac;

  // output
  DEFINE
  {"\n  ".join([f"ins_ready_{n} := index = {select_type.format_constant(n)} & index_valid & inner_tehb.ins_ready & ins_valid_{n} | !ins_valid_{n};" for n in range(size)])}
  index_ready := !index_valid | tehb_ins_valid & inner_tehb.ins_ready;
  outs_valid := inner_tehb.outs_valid;

{generate_buffer(f"{name}__tehb_dataless", {"slots": 1, "timing": "R: 1", "data_type": HANSHAKE_CONTROL_TYPE.mlir_type})}
"""


def _generate_mux(name, size, data_type, select_type):
  return f"""
MODULE {name}({", ".join([f"ins_{n}, ins_valid_{n}" for n in range(size)])}, index, index_valid, outs_ready)
  VAR
  inner_tehb : {name}__tehb(tehb_ins, tehb_ins_valid, outs_ready);

  DEFINE
  tehb_ins := case
    {"\n    ".join([f"index = {select_type.format_constant(n)} & index_valid & ins_valid_{n} : ins_{n};" for n in range(size)])}
    TRUE : ins_0;
  esac;
  tehb_ins_valid := case
    {"\n    ".join([f"index = {select_type.format_constant(n)} : index_valid & ins_valid_{n} | !ins_valid_{n};" for n in range(size)])}
    TRUE : FALSE;
  esac;

  // output
  DEFINE
  {"\n  ".join([f"ins_ready_{n} := index = {select_type.format_constant(n)} & index_valid & inner_tehb.ins_ready & ins_valid_{n} | !ins_valid_{n};" for n in range(size)])}
  index_ready := !index_valid | tehb_ins_valid & inner_tehb.ins_ready;
  outs_valid := inner_tehb.outs_valid;
  outs := inner_tehb.outs;

{generate_buffer(f"{name}__tehb", {"slots": 1, "timing": "R: 1", "data_type": data_type.mlir_type})}
"""