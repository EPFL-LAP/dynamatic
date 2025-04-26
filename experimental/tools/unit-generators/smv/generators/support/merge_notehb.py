from generators.support.utils import *


def generate_merge_notehb(name, params):
  size = params[ATTR_SIZE]
  data_type = SmvScalarType(params[ATTR_DATA_TYPE])

  if data_type.bitwidth == 0:
    return _generate_merge_notehb_dataless(name, size)
  else:
    return _generate_merge_notehb(name, size, data_type)


def _generate_merge_notehb_dataless(name, size):
  return f"""
MODULE {name}({", ".join([f"ins_valid_{n}" for n in range(size)])}, outs_ready)

  DEFINE
  one_valid := {' | '.join([f'ins_valid_{i}' for i in range(size)])};

  // output
  DEFINE
  {"\n  ".join([f"ins_ready_{n} := ins_valid_{n} & outs_ready;" for n in range(size)])}
  outs_valid := one_valid;
"""


def _generate_merge_notehb(name, size, data_type):
  return f"""
MODULE {name}({", ".join([f"ins_{n}" for n in range(size)])}, {", ".join([f"ins_valid_{n}" for n in range(size)])}, outs_ready)

  DEFINE
  one_valid := {' | '.join([f'ins_valid_{i}' for i in range(size)])};
  data := case
    {"\n    ".join([f"ins_valid_{n} : ins_{n};" for n in range(size)])}
    TRUE : FALSE;
  esac;

  // output
  DEFINE
  {"\n  ".join([f"ins_ready_{n} := ins_valid_{n} & outs_ready;" for n in range(size)])}
  outs_valid := one_valid;
  outs := data;
"""
