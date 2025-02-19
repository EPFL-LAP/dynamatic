from generators.support.eager_fork_register_block import (
    generate_eager_fork_register_block,
)
from generators.support.utils import *


def generate_fork(name, params):
  size = params[ATTR_SIZE]
  data_type = SmvScalarType(params[ATTR_PORT_TYPES]["ins"])

  if data_type.bitwidth == 0:
    return _generate_fork_dataless(name, size)
  else:
    return _generate_fork(name, size, data_type)


def _generate_fork_dataless(name, size):
  return f"""
MODULE {name}(ins_valid, {", ".join([f"outs_ready_{n}" for n in range(size)])})
  {"\n    ".join([f"VAR inner_reg_block_{n} : {name}__eager_fork_register_block(ins_valid, outs_ready_{n}, backpressure);" for n in range(size)])}

  DEFINE
  any_block_stop := {" | ".join([f"inner_reg_block_{n}.block_stop" for n in range(size)])};
  backpressure := ins_valid & any_block_stop;

  // output
  DEFINE
  ins_ready := !any_block_stop;
  {"\n    ".join([f"outs_valid_{n} := inner_reg_block_{n}.outs_valid;" for n in range(size)])}

{generate_eager_fork_register_block(f"{name}__eager_fork_register_block")}
"""


def _generate_fork(name, size, data_type):
  return f"""
MODULE {name}(ins, ins_valid, {", ".join([f"outs_ready_{n}" for n in range(size)])})
  VAR
  inner_fork : {name}__fork_dataless(ins_valid, {", ".join([f"outs_ready_{n}" for n in range(size)])});

  //output
  DEFINE
  ins_ready := inner_fork.ins_ready;
  {"\n    ".join([f"outs_{n} := ins;" for n in range(size)])}
  {"\n    ".join([f"outs_valid_{n} := inner_fork.outs_valid_{n};" for n in range(size)])}

{_generate_fork_dataless(f"{name}__fork_dataless", size)}
"""
