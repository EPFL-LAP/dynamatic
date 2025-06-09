from generators.support.tehb import generate_tehb
from generators.handshake.fork import generate_fork
from generators.handshake.merge import generate_merge
from generators.support.utils import *


def generate_control_merge(name, params):
    size = params[ATTR_SIZE]
    data_type = SmvScalarType(params[ATTR_DATA_BITWIDTH])
    index_type = SmvScalarType(params[ATTR_INDEX_BITWIDTH])

    if data_type.bitwidth == 0:
        return _generate_control_merge_dataless(name, size, index_type)
    else:
        return _generate_control_merge(name, size, index_type, data_type)


def _generate_control_merge_dataless(name, size, index_type):
    return f"""
MODULE {name}({", ".join([f"ins_{n}_valid" for n in range(size)])}, outs_ready, index_ready)
  VAR
  inner_tehb : {name}__tehb(index_in, inner_merge.outs_valid, inner_fork.ins_ready);
  inner_merge : {name}__merge_dataless({", ".join([f"ins_{n}_valid" for n in range(size)])}, inner_tehb.ins_ready);
  inner_fork : {name}__fork_dataless(inner_tehb.outs_valid, outs_ready, index_ready);

  DEFINE
  index_in := case
    {"\n    ".join([f"ins_{n}_valid = TRUE : {index_type.format_constant(n)};" for n in range(size)])}
    TRUE: {index_type.format_constant(0)};
  esac;

  -- output
  DEFINE
  {"\n  ".join([f"ins_{n}_ready := inner_merge.ins_{n}_ready;" for n in range(size)])}
  outs_valid := inner_fork.outs_0_valid;
  index_valid := inner_fork.outs_1_valid;
  index := inner_tehb.outs;

{generate_merge(f"{name}__merge_dataless", {ATTR_SIZE: size, ATTR_BITWIDTH: 0})}
{generate_tehb(f"{name}__tehb", {ATTR_BITWIDTH: index_type.bitwidth})}
{generate_fork(f"{name}__fork_dataless", {ATTR_SIZE: 2, ATTR_BITWIDTH: 0})}
"""


def _generate_control_merge(name, size, index_type, data_type):
    return f"""
  MODULE {name}({", ".join([f"ins_{n}" for n in range(size)])}, {", ".join([f"ins_{n}_valid" for n in range(size)])}, outs_ready, index_ready)
  VAR
  inner_control_merge : {name}__control_merge_dataless({", ".join([f"ins_{n}_valid" for n in range(size)])}, outs_ready, index_ready);

  DEFINE
  data := case
    {"\n    ".join([f"index = {index_type.format_constant(n)}: ins_{n};" for n in range(size)])}
    TRUE: {data_type.format_constant(0)};
  esac;

  -- output
  DEFINE
  {"\n  ".join([f"ins_{n}_ready := inner_control_merge.ins_{n}_ready;" for n in range(size)])}
  outs_valid := inner_control_merge.outs_valid;
  index_valid := inner_control_merge.index_valid;
  outs := data;
  index := inner_control_merge.index;

{_generate_control_merge_dataless(f"{name}__control_merge_dataless", size, index_type)}
"""
