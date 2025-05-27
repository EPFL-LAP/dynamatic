from generators.support.utils import *


def generate_lazy_fork(name, params):
    size = params[ATTR_SIZE]
    data_type = SmvScalarType(params[ATTR_BITWIDTH])

    if data_type.bitwidth == 0:
        return _generate_lazy_fork_dataless(name, size)
    else:
        return _generate_lazy_fork(name, size, data_type)


def _generate_lazy_fork_dataless(name, size):
    return f"""
MODULE {name}(ins_valid, {", ".join([f"outs_{n}_ready" for n in range(size)])})

  DEFINE
  all_ready := {" & ".join([f"outs_{n}_ready" for n in range(size)])};
  {"\n  ".join([f"others_ready_{n} := {" & ".join([f"outs_{i}_ready" for i in range(size) if i != n])};" for n in range(size)])}


  -- output
  DEFINE
  ins_ready := all_ready;
  {"\n  ".join([f"outs_{n}_valid := ins_valid & others_ready_{n};" for n in range(size)])}
"""


def _generate_lazy_fork(name, size, data_type):
    return f"""
MODULE {name}(ins, ins_valid, {", ".join([f"outs_{n}_ready" for n in range(size)])})
  VAR
  inner_lazy_fork : {name}__lazy_fork_dataless(ins_valid, {", ".join([f"outs_{n}_ready" for n in range(size)])});

  //output
  DEFINE
  ins_ready := inner_lazy_fork.ins_ready;
  {"\n  ".join([f"outs_{n}_valid := inner_lazy_fork.outs_{n}_valid;" for n in range(size)])}
  {"\n  ".join([f"outs_{n} := ins;" for n in range(size)])}

{_generate_lazy_fork_dataless(f"{name}__lazy_fork_dataless", size)}
"""
