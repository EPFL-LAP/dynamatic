from generators.support.utils import *


def generate_join(name, params):
    size = params[ATTR_SIZE]

    return _generate_join(name, size)


def _generate_join(name, size):
    return f"""
MODULE {name}({", ".join([f"ins_{n}_valid" for n in range(size)])}, outs_ready)

  DEFINE
  all_valid := {" & ".join([f"ins_{n}_valid" for n in range(size)])};

  -- output
  DEFINE
  {"\n  ".join([f"ins_{n}_ready := outs_ready & {" & ".join([f"ins_{m}_valid" for m in range(size) if m != n])};" for n in range(size)])}
  outs_valid := all_valid;
"""
