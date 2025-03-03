from generators.support.elastic_fifo_inner import generate_elastic_fifo_inner
from generators.support.utils import *


def generate_buffer(name, params):
    match_r = re.search(r"R: (\d+)", params[ATTR_TIMING])
    timing_r = False if not match_r else bool(match_r[0])
    match_d = re.search(r"D: (\d+)", params[ATTR_TIMING])
    timing_d = False if not match_d else bool(match_d[0])
    match_v = re.search(r"V: (\d+)", params[ATTR_TIMING])
    timing_v = False if not match_v else bool(match_v[0])
    transparent = timing_r and not (timing_d or timing_v)

    slots = params[ATTR_SLOTS] if ATTR_SLOTS in params else 1
    data_type = SmvScalarType(params[ATTR_PORT_TYPES]["outs"])

    if transparent and slots > 1 and data_type.bitwidth == 0:
        return _generate_tfifo_dataless(name, slots)
    elif transparent and slots > 1 and data_type.bitwidth != 0:
        return _generate_tfifo(name, slots, data_type)
    elif transparent and slots == 1 and data_type.bitwidth == 0:
        return _generate_tehb_dataless(name)
    elif transparent and slots == 1 and data_type.bitwidth != 0:
        return _generate_tehb(name, data_type)
    elif not transparent and slots > 1 and data_type.bitwidth == 0:
        return _generate_ofifo_dataless(name, slots)
    elif not transparent and slots > 1 and data_type.bitwidth != 0:
        return _generate_ofifo(name, slots, data_type)
    elif not transparent and slots == 1 and data_type.bitwidth == 0:
        return _generate_oehb_dataless(name)
    elif not transparent and slots == 1 and data_type.bitwidth != 0:
        return _generate_oehb(name, data_type)
    else:
        raise ValueError(f"Buffer implementation nt found")


def _generate_oehb_dataless(name):
    return f"""
MODULE {name} (ins_valid, outs_ready)
  VAR
  outs_valid_i : boolean;

  ASSIGN
  init(outs_valid_i) := FALSE;
  next(outs_valid_i) := ins_valid | (outs_valid_i & !outs_ready);

  // output
  DEFINE
  ins_ready := !outs_valid_i | outs_ready;
  outs_valid := outs_valid_i;
"""


def _generate_oehb(name, data_type):
    return f"""
MODULE {name} (ins, ins_valid, outs_ready)
  VAR
  inner_oehb : {name}__oehb_dataless(ins_valid, outs_ready);
  data : {data_type};

  ASSIGN
  init(data) := {data_type.format_constant(0)};
  next(data) := case
    ins_ready & ins_valid : ins;
    TRUE : data;
  esac;
    
  // output
  DEFINE
  ins_ready := inner_oehb.ins_ready;
  outs_valid := inner_oehb.outs_valid;
  outs := data;

{_generate_oehb_dataless(f"{name}__oehb_dataless")}
"""


def _generate_ofifo_dataless(name, slots):
    return f"""
MODULE {name} (ins_valid, outs_ready)
  VAR
  inner_tehb : {name}__tehb_dataless(ins_valid, inner_elastic_fifo.ins_ready);
  inner_elastic_fifo : {name}__elastic_fifo_inner_dataless(inner_tehb.outs_valid, outs_ready);

  // output
  DEFINE
  ins_ready := inner_tehb.ins_ready;
  outs_valid := inner_elastic_fifo.outs_valid;

{_generate_tehb_dataless(f"{name}__tehb_dataless")}
{generate_elastic_fifo_inner(f"{name}__elastic_fifo_inner_dataless", {ATTR_SLOTS: slots, ATTR_DATA_TYPE: HANDSHAKE_CONTROL_TYPE.mlir_type})}
"""


def _generate_ofifo(name, slots, data_type):
    return f"""
MODULE {name} (ins, ins_valid, outs_ready)
  VAR
  inner_tehb : {name}__tehb(ins, ins_valid, inner_elastic_fifo.ins_ready);
  inner_elastic_fifo : {name}__elastic_fifo_inner(inner_tehb.outs, inner_tehb.outs_valid, outs_ready);

  // output
  DEFINE
  ins_ready := inner_tehb.ins_ready;
  outs_valid := inner_elastic_fifo.outs_valid;
  outs := inner_elastic_fifo.outs;

{_generate_tehb(f"{name}__tehb_dataless", data_type)}
{generate_elastic_fifo_inner(f"{name}__elastic_fifo_inner_dataless", {ATTR_SLOTS: slots, ATTR_DATA_TYPE: data_type.mlir_type})}
"""


def _generate_tehb_dataless(name):
    return f"""
MODULE {name}(ins_valid, outs_ready)
  VAR
  full : boolean;

  ASSIGN
  init(full) := FALSE;
  next(full) := outs_valid & !outs_ready;

  // output
  DEFINE
  ins_ready := !full;
  outs_valid := ins_valid | full;
"""


def _generate_tehb(name, data_type):
    return f"""
MODULE {name}(ins, ins_valid, outs_ready)
  VAR
  inner_tehb : {name}__tehb_dataless(ins_valid, outs_ready);
  data : {data_type};

  ASSIGN
  init(data) := {data_type.format_constant(0)};
  next(data) := ins_ready & ins_valid & !outs_ready ? ins : data;

  // output
  DEFINE
  ins_ready := inner_tehb.ins_ready;
  outs_valid := inner_tehb.outs_valid;
  outs := inner_tehb.full ? data : ins;

{_generate_tehb_dataless(f"{name}__tehb_dataless")}
"""


def _generate_tfifo_dataless(name, slots):
    return f"""
MODULE {name} (ins_valid, outs_ready)
  VAR
  inner_elastic_fifo : {name}__elastic_fifo_inner_dataless(fifo_valid, fifo_ready);

  DEFINE
  fifo_valid := ins_valid & (!outs_ready | inner_elastic_fifo.outs_valid);
  fifo_ready := outs_ready;

  // output
  DEFINE
  ins_ready := inner_elastic_fifo.ins_ready | outs_ready;
  outs_valid := ins_valid | inner_elastic_fifo.outs_valid;

{generate_elastic_fifo_inner(f"{name}__elastic_fifo_inner_dataless", {ATTR_SLOTS: slots, ATTR_DATA_TYPE: HANDSHAKE_CONTROL_TYPE.mlir_type})}
"""


def _generate_tfifo(name, slots, data_type):
    return f"""
MODULE {name} (ins, ins_valid, outs_ready)
  VAR
  inner_elastic_fifo : {name}__elastic_fifo_inner(ins, fifo_valid, fifo_ready);

  DEFINE
  fifo_valid := ins_valid & (!outs_ready | inner_elastic_fifo.outs_valid);
  fifo_ready := outs_ready;

  // output
  DEFINE
  ins_ready := inner_elastic_fifo.ins_ready | outs_ready;
  outs_valid := ins_valid | inner_elastic_fifo.outs_valid;
  outs := inner_elastic_fifo.outs_valid ? inner_elastic_fifo.outs : ins;

{generate_elastic_fifo_inner(f"{name}__elastic_fifo_inner", {ATTR_SLOTS: slots, ATTR_DATA_TYPE: data_type.mlir_type})}
"""
