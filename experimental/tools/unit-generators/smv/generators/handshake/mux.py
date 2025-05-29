from generators.support.tehb import generate_tehb
from generators.support.utils import *


def generate_mux(name, params):
    size = params[ATTR_SIZE]
    data_type = SmvScalarType(params[ATTR_DATA_BITWIDTH])
    select_type = SmvScalarType(params[ATTR_INDEX_BITWIDTH])

    if data_type.bitwidth == 0:
        return _generate_mux_dataless(name, size, select_type)
    else:
        return _generate_mux(name, size, data_type, select_type)


def _generate_mux_dataless(name, size, select_type):
    return f"""
MODULE {name}(index, index_valid, {", ".join([f"ins_{n}_valid" for n in range(size)])}, outs_ready)
  VAR
  inner_tehb : {name}__tehb_dataless(tehb_ins_valid, outs_ready);

  DEFINE
  tehb_ins_valid := case
    {"\n    ".join([f"index = {select_type.format_constant(n)} : index_valid & ins_{n}_valid;" for n in range(size)])}
    TRUE : FALSE;
  esac;

  -- output
  DEFINE
  {"\n  ".join([f"ins_{n}_ready := index = {select_type.format_constant(n)} & index_valid & inner_tehb.ins_ready & ins_{n}_valid | !ins_{n}_valid;" for n in range(size)])}
  index_ready := !index_valid | tehb_ins_valid & inner_tehb.ins_ready;
  outs_valid := inner_tehb.outs_valid;

{generate_tehb(f"{name}__tehb_dataless", {ATTR_BITWIDTH: 0})}
"""


def _generate_mux(name, size, data_type, select_type):
    return f"""
MODULE {name}(index, index_valid, {", ".join([f"ins_{n}" for n in range(size)])}, {", ".join([f"ins_{n}_valid" for n in range(size)])}, outs_ready)
  VAR
  inner_tehb : {name}__tehb(tehb_ins, tehb_ins_valid, outs_ready);

  DEFINE
  tehb_ins := case
    {"\n    ".join([f"index = {select_type.format_constant(n)} & index_valid & ins_{n}_valid : ins_{n};" for n in range(size)])}
    TRUE : ins_0;
  esac;
  tehb_ins_valid := case
    {"\n    ".join([f"index = {select_type.format_constant(n)} : index_valid & ins_{n}_valid;" for n in range(size)])}
    TRUE : FALSE;
  esac;

  -- output
  DEFINE
  {"\n  ".join([f"ins_{n}_ready := index = {select_type.format_constant(n)} & index_valid & inner_tehb.ins_ready & ins_{n}_valid | !ins_{n}_valid;" for n in range(size)])}
  index_ready := !index_valid | tehb_ins_valid & inner_tehb.ins_ready;
  outs_valid := inner_tehb.outs_valid;
  outs := inner_tehb.outs;

{generate_tehb(f"{name}__tehb", {ATTR_BITWIDTH: data_type.bitwidth})}
"""
