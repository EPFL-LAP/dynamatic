from generators.support.utils import *


def generate_transfer_control(name, params):
    data_type = SmvScalarType(params[ATTR_BITWIDTH])

    if data_type.bitwidth == 0:
        return _generate_transfer_control_dataless(name)
    else:
        return _generate_transfer_control(name, data_type)


def _generate_transfer_control_dataless(name):
    return f"""
MODULE {name}(ins_valid, ctrl, ctrl_valid, outs_ready)
  DEFINE
  // output
  outs_valid := ins_valid & ctrl_valid & ctrl;
  ins_ready := outs_ready & ctrl_valid & ctrl;
  ctrl_ready := outs_ready & ins_valid & ctrl;
"""


def _generate_transfer_control(name, _):
    return f"""
MODULE {name}(ins, ins_valid, ctrl, ctrl_valid, outs_ready)
  VAR inner_tc : {name}__tc_dataless(ins_valid, ctrl, ctrl_valid, outs_ready);
  DEFINE
  // output
  outs := ins;
  outs_valid := inner_tc.outs_valid;
  ins_ready := inner_tc.ins_ready;
  ctrl_ready := inner_tc.ctrl_ready;
{_generate_transfer_control_dataless(f"{name}__tc_dataless")}
"""
