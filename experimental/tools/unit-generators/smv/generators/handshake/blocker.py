from generators.support.utils import *


def generate_blocker(name, params):
    data_type = SmvScalarType(params[ATTR_BITWIDTH])

    if data_type.bitwidth == 0:
        return _generate_blocker_dataless(name)
    else:
        return _generate_blocker(name, data_type)


def _generate_blocker_dataless(name):
    return f"""
MODULE {name}(ins_valid, ctrl_valid, outs_ready)
  DEFINE
  // output
  outs_valid := ins_valid & ctrl_valid;
  ins_ready := outs_ready & ctrl_valid;
  ctrl_ready := outs_ready & ins_valid;
"""


def _generate_blocker(name, _):
    return f"""
MODULE {name}(ins, ins_valid, ctrl_valid, outs_ready)
  VAR inner_blocker : {name}__tc_dataless(ins_valid, ctrl_valid, outs_ready);
  DEFINE
  // output
  outs := ins;
  outs_valid := inner_blocker.outs_valid;
  ins_ready := inner_blocker.ins_ready;
  ctrl_ready := inner_blocker.ctrl_ready;
{_generate_blocker_dataless(f"{name}__tc_dataless")}
"""
