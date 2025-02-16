from generators.support.utils import *


def generate_lazy_fork(name, params):
  size = params[ATTR_SIZE]
  data_type = SmvScalarType(params[ATTR_DATA_TYPE])

  if data_type.bitwidth == 0:
    return _generate_lazy_fork_dataless(name, size)
  else:
    return _generate_lazy_fork(name, size, data_type)


def _generate_lazy_fork_dataless(name, size):
  return f"""
MODULE {name}(ins_valid, {", ".join([f"outs_ready_{n}" for n in range(size)])})

  DEFINE
  all_ready := {" & ".join([f"outs_ready_{n}" for n in range(size)])};

  // output
  DEFINE
  ins_ready := all_ready;
  {"\n  ".join([f"outs_valid_{n} := ins_valid & all_ready;" for n in range(size)])}
"""


def _generate_lazy_fork(name, size, data_type):
  return f"""
MODULE {name}(ins, ins_valid, {", ".join([f"outs_ready_{n}" for n in range(size)])})
  VAR
  inner_lazy_fork : {name}__lazy_fork_dataless(ins_valid, {", ".join([f"outs_ready_{n}" for n in range(size)])});

  //output
  DEFINE
  ins_ready := inner_lazy_fork.ins_ready;
  {"\n  ".join([f"outs_valid_{n} := inner_lazy_fork.outs_valid_{n};" for n in range(size)])}
  {"\n  ".join([f"outs_{n} := ins;" for n in range(size)])}

{_generate_lazy_fork_dataless(f"{name}__lazy_fork_dataless", size)}
"""