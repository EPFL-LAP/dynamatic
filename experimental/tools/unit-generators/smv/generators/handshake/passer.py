from generators.handshake.join import generate_join
from generators.support.utils import *


def generate_passer(name, params):
    data_type = SmvScalarType(params[ATTR_BITWIDTH])

    if data_type.bitwidth == 0:
        return _generate_passer_dataless(name)
    else:
        return _generate_passer(name, data_type)


def _generate_passer_dataless(name):
    return f"""
MODULE {name}(data_valid, ctrl, ctrl_valid, result_ready)
  VAR
  inner_join : {name}__join(data_valid, ctrl_valid, branch_ready);
  DEFINE
  branch_ready := !ctrl | result_ready;

  // output
  data_ready := inner_join.ins_0_ready;
  ctrl_ready := inner_join.ins_1_ready;
  result_valid := ctrl & inner_join.outs_valid;

{generate_join(f"{name}__join", {ATTR_SIZE: 2})}
"""


def _generate_passer(name, _):
    return f"""
MODULE {name}(data, data_valid, ctrl, ctrl_valid, result_ready)
  VAR
  inner_passer : {name}__passer_dataless(data_valid, ctrl, ctrl_valid, result_ready);

  DEFINE
  result := data;
  data_ready := inner_passer.data_ready;
  ctrl_ready := inner_passer.ctrl_ready;
  result_valid := inner_passer.result_valid;

{_generate_passer_dataless(f"{name}__passer_dataless")}
"""
