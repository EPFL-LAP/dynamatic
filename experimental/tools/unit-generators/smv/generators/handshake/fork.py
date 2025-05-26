from generators.support.eager_fork_register_block import (
    generate_eager_fork_register_block,
)
from generators.support.utils import *


def generate_fork(name, params):
    size = params[ATTR_SIZE]
    data_type = SmvScalarType(params[ATTR_BITWIDTH])

    if data_type.bitwidth == 0:
        return _generate_fork_dataless(name, size)
    else:
        return _generate_fork(name, size, data_type)


def _generate_fork_dataless(name, size):
    return f"""
MODULE {name}(ins_valid, {", ".join([f"outs_{n}_ready" for n in range(size)])})
  {"\n    ".join([f"VAR inner_reg_block_{n} : {name}__eager_fork_register_block(ins_valid, outs_{n}_ready, backpressure);" for n in range(size)])}

  DEFINE
  any_block_stop := {" | ".join([f"inner_reg_block_{n}.block_stop" for n in range(size)])};
  backpressure := ins_valid & any_block_stop;

  -- output
  DEFINE
  ins_ready := !any_block_stop;
  {"\n  ".join([f"outs_{n}_valid := inner_reg_block_{n}.outs_valid;" for n in range(size)])}

{generate_eager_fork_register_block(f"{name}__eager_fork_register_block")}
"""


def _generate_fork(name, size, data_type):
    return f"""
MODULE {name}(ins, ins_valid, {", ".join([f"outs_{n}_ready" for n in range(size)])})
  VAR
  inner_fork : {name}__fork_dataless(ins_valid, {", ".join([f"outs_{n}_ready" for n in range(size)])});

  //output
  DEFINE
  ins_ready := inner_fork.ins_ready;
  {"\n  ".join([f"outs_{n} := ins;" for n in range(size)])}
  {"\n  ".join([f"outs_{n}_valid := inner_fork.outs_{n}_valid;" for n in range(size)])}

{_generate_fork_dataless(f"{name}__fork_dataless", size)}
"""
