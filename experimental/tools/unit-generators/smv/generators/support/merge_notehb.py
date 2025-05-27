from generators.support.utils import *


def generate_merge_notehb(name, params):
    size = params[ATTR_SIZE]
    data_type = SmvScalarType(params[ATTR_BITWIDTH])

    if data_type.bitwidth == 0:
        return _generate_merge_notehb_dataless(name, size)
    else:
        return _generate_merge_notehb(name, size, data_type)


def _generate_merge_notehb_dataless(name, size):
    return f"""
MODULE {name}({", ".join([f"ins_{n}_valid" for n in range(size)])}, outs_ready)
  
  DEFINE
  one_valid := {' | '.join([f'ins_{i}_valid' for i in range(size)])};
  in_ins_0_ready := ins_0_valid ? outs_ready : FALSE;
  {"\n  ".join([f"in_ins_{n + 1}_ready := (ins_{n + 1}_valid & !({' | '.join([f'in_ins_{i}_ready' for i in range(n + 1)])})) ? outs_ready : FALSE;" for n in range(size - 1)])}


  -- output
  DEFINE
  {"\n  ".join([f"ins_{n}_ready := in_ins_{n}_ready;" for n in range(size)])}
  outs_valid := one_valid;
"""


def _generate_merge_notehb(name, size, data_type):
    return f"""
MODULE {name}({", ".join([f"ins_{n}" for n in range(size)])}, {", ".join([f"ins_{n}_valid" for n in range(size)])}, outs_ready)

  DEFINE
  one_valid := {' | '.join([f'ins_{i}_valid' for i in range(size)])};
  in_ins_0_ready := ins_0_valid ? outs_ready : FALSE;
  {"\n  ".join([f"in_ins_{n + 1}_ready := (ins_{n + 1}_valid & !({' | '.join([f'in_ins_{i}_ready' for i in range(n + 1)])})) ? outs_ready : FALSE;" for n in range(size - 1)])}
  data := case
    {"\n    ".join([f"ins_{n}_valid : ins_{n};" for n in range(size)])}
    TRUE : {data_type.format_constant(0)};
  esac;

  -- output
  DEFINE
  {"\n  ".join([f"ins_{n}_ready := in_ins_{n}_ready;" for n in range(size)])}
  outs_valid := one_valid;
  outs := data;
"""
