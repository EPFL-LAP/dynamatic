from generators.handshake.join import generate_join
from generators.support.utils import *


def generate_cond_br(name, params):
    data_type = SmvScalarType(params[ATTR_BITWIDTH])

    if data_type.bitwidth == 0:
        return _generate_cond_br_dataless(name)
    else:
        return _generate_cond_br(name, data_type)


def _generate_cond_br_dataless(name):
    return f"""
MODULE {name}(condition, condition_valid, data_valid, trueOut_ready, falseOut_ready)
  VAR
  inner_join : {name}__join(data_valid, condition_valid, branch_ready);

  DEFINE
  branch_ready := (falseOut_ready & !condition) | (trueOut_ready & condition);

  -- output
  DEFINE
  data_ready := inner_join.ins_0_ready;
  condition_ready := inner_join.ins_1_ready;
  trueOut_valid := condition & inner_join.outs_valid;
  falseOut_valid := !condition & inner_join.outs_valid;

{generate_join(f"{name}__join", {ATTR_SIZE: 2})}
"""


def _generate_cond_br(name, data_type):
    return f"""
MODULE {name}(condition, condition_valid, data, data_valid, trueOut_ready, falseOut_ready)
  VAR
  inner_br : {name}__cond_br_dataless(condition, condition_valid, data_valid, trueOut_ready, falseOut_ready);

  -- output
  DEFINE
  data_ready := inner_br.data_ready;
  condition_ready := inner_br.condition_ready;
  trueOut_valid := inner_br.trueOut_valid;
  falseOut_valid := inner_br.falseOut_valid;
  trueOut := data;
  falseOut := data;

{_generate_cond_br_dataless(f"{name}__cond_br_dataless")}
"""
