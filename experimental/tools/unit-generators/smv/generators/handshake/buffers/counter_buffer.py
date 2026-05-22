from generators.support.utils import *

def generate_counter_buffer(name, params):
    bitwidth = params["bitwidth"]
    dv_latency = int(params["dv_latency"])
    extra_signals = params.get("extra_signals", None)
    data_type = SmvScalarType(params[ATTR_BITWIDTH])

    if extra_signals:
        return _generate_counter_buffer_signal_manager(name, dv_latency, bitwidth, extra_signals)
    if bitwidth == 0:
        return _generate_counter_buffer_dataless(name, dv_latency)
    else:
        return _generate_counter_buffer(name, dv_latency, data_type)

def _counter_width(dv_latency):
    return 1 if dv_latency <= 1 else (dv_latency - 1).bit_length()

def _delay_counter_assignment(dv_latency):

    assert dv_latency > 0

    if (dv_latency == 1):
        return ""

    ret = "  init(delayCnt) := {dv_latency - 1};"
    ret += "  next(delayCnt) := case\n"
    ret += f"  !occupied & ins_valid : {dv_latency - 1};\n"
    for i in range(dv_latency):
        if i > 0:
            ret += f"  occupied & delayCnt = {i} : {i-1};\n"
    ret += f"  occupied & delayCnt = 0 : (outs_ready & ins_valid ? {dv_latency - 1} : delayCnt);\n"
    ret += "esac;"
    return ret


def _generate_counter_buffer_dataless(name, dv_latency):
    cnt_w = _counter_width(dv_latency)

    return f"""
MODULE {name} (ins_valid, outs_ready)
  VAR
  occupied     : boolean;
  delayCnt     : {dv_latency - 1}..0;

  ASSIGN
  init(occupied) := FALSE;
  next(occupied) := case
    !occupied : (ins_valid ? TRUE : occupied);
    outs_valid & outs_ready : ins_valid;
    TRUE : occupied;
  esac;

{_delay_counter_assignment(dv_latency)}

  -- output
  DEFINE
  ins_ready := (!occupied) | (outs_valid & outs_ready);
  outs_valid := occupied & (delayCnt = 0);
"""


def _generate_counter_buffer(name, dv_latency, data_type):
    cnt_w = _counter_width(dv_latency)

    return f"""
MODULE {name} (ins, ins_valid, outs_ready)
  VAR
  occupied     : boolean;
  delayCnt     : {dv_latency - 1}..0;
  data : {data_type};

  ASSIGN
  init(occupied) := FALSE;
  next(occupied) := case
    !occupied : (ins_valid ? TRUE : occupied);
    outs_valid & outs_ready : ins_valid;
    TRUE : occupied;
  esac;

  ASSIGN
{_delay_counter_assignment(dv_latency)}

  -- output
  DEFINE
  ins_ready := (!occupied) | (outs_valid & outs_ready);
  outs_valid := occupied & (delayCnt = 0);
  outs := data;

  ASSIGN
  init(data) := {data_type.format_constant(0)};
  next(data) := case
    ins_ready & ins_valid : ins;
    TRUE : data;
  esac;

"""


