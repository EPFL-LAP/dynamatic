from generators.support.merge_notehb import (
    generate_merge_notehb,
)
from generators.support.tehb import generate_tehb
from generators.support.utils import *


def generate_merge(name, params):
    size = params[ATTR_SIZE]
    data_type = SmvScalarType(params[ATTR_BITWIDTH])

    if data_type.bitwidth == 0:
        return _generate_merge_dataless(name, size)
    else:
        return _generate_merge(name, size, data_type)


def _generate_merge_dataless(name, size):
    return f"""
MODULE {name}({", ".join([f"ins_{n}_valid" for n in range(size)])}, outs_ready)
  VAR outs_valid : boolean;
  {"\n  ".join([f"VAR ins_{n}_ready : boolean;" for n in range(size)])}

  ASSIGN
  init(outs_valid) := FALSE;
  {"\n  ".join([f"init(ins_{n}_ready) := FALSE;" for n in range(size)])}  

  DEFINE
  one_valid := {' | '.join([f'ins_{i}_valid' for i in range(size)])};
  in_ins_0_ready := ins_0_valid ? outs_ready : FALSE;
  {"\n  ".join([f"in_ins_{n + 1}_ready := (ins_{n + 1}_valid & !({' | '.join([f'in_ins_{i}_ready' for i in range(n + 1)])})) ? outs_ready : FALSE;" for n in range(size - 1)])}


  -- output
  ASSIGN
  {"\n  ".join([f"next(ins_{n}_ready) := in_ins_{n}_ready;" for n in range(size)])}
  next(outs_valid) := one_valid;
"""


def _generate_merge(name, size, data_type):
    return f"""
MODULE {name}({", ".join([f"ins_{n}_valid" for n in range(size)])}, outs_ready)
  VAR outs_valid : boolean;
  VAR outs : {data_type};
  {"\n  ".join([f"VAR ins_{n}_ready : boolean;" for n in range(size)])}

  ASSIGN
  init(outs_valid) := FALSE;
  init(outs) := {data_type.format_constant(0)};
  {"\n  ".join([f"init(ins_{n}_ready) := FALSE;" for n in range(size)])}  

  DEFINE
  one_valid := {' | '.join([f'ins_{i}_valid' for i in range(size)])};
  in_ins_0_ready := ins_0_valid ? outs_ready : FALSE;
  {"\n  ".join([f"in_ins_{n + 1}_ready := (ins_{n + 1}_valid & !({' | '.join([f'in_ins_{i}_ready' for i in range(n + 1)])})) ? outs_ready : FALSE;" for n in range(size - 1)])}


  -- output
  ASSIGN
  next(outs_valid) := one_valid;
  next(outs) := ins;
  {"\n  ".join([f"next(ins_{n}_ready) := in_ins_{n}_ready;" for n in range(size)])}
"""
