from generators.support.oehb import generate_oehb
from generators.support.utils import *


def generate_delay_buffer(name, params):
    latency = params[ATTR_LATENCY]

    if latency == 0:
        return _generate_no_lat_delay_buffer(name)
    elif latency == 1:
        return _generate_single_delay_buffer(name)
    else:
        return _generate_delay_buffer(name, latency)


def _generate_no_lat_delay_buffer(name):
    return f"""
MODULE {name}(ins_valid, outs_ready)

  -- output
  DEFINE ins_ready := outs_ready;
  DEFINE outs_valid := ins_valid;
"""


def _generate_single_delay_buffer(name):
    return f"""
MODULE {name}(ins_valid, outs_ready)
  VAR inner_oehb : {name}__oehb_dataless(ins, ins_valid, outs_ready);

  -- output
  DEFINE ins_ready := inner_oehb.ins_ready;
  DEFINE outs_valid := inner_oehb.outs_valid;

{generate_oehb(f"{name}__oehb_dataless", {ATTR_BITWIDTH: 0})}
"""


def _generate_delay_buffer(name, latency):
    return f"""
MODULE {name}(ins_valid, outs_ready)
  VAR inner_oehb : {name}__oehb_dataless(v{latency - 1}, outs_ready);
  {"\n  ".join([f"VAR v{n + 1} : boolean;" for n in range(latency - 1)])}

  DEFINE v0 := ins_valid;

  {"\n  ".join([f"""ASSIGN init(v{n + 1}) := FALSE;
  ASSIGN next(v{n + 1}) := inner_oehb.ins_ready ? v{n} : v{n + 1};
""" for n in range(latency - 1)])}

  -- output
  DEFINE ins_ready := inner_oehb.ins_ready;
  DEFINE outs_valid := inner_oehb.outs_valid;

{generate_oehb(f"{name}__oehb_dataless", {ATTR_BITWIDTH: 0})}
"""
