from generators.support.utils import *


def generate_elastic_fifo_inner(name, params):
    slots = params[ATTR_SLOTS] if ATTR_SLOTS in params else 1
    data_type = SmvScalarType(params[ATTR_BITWIDTH])

    if slots > 2:
        # This model cascades one_slot_break_none and its total reachable states scales linearly withs the size of the FIFO,
        # instead of scaling quadratically like the head-tail based model.
        if data_type.bitwidth == 0:
            return _generate_slot_based_elastic_fifo_inner_dataless(name, slots)
        else:
            return _generate_slot_based_elastic_fifo_inner(name, slots, data_type)
    else:
        if data_type.bitwidth == 0:
            return _generate_elastic_fifo_inner_dataless(name, slots)
        else:
            return _generate_elastic_fifo_inner(name, slots, data_type)


def _generate_one_slot_break_none_dataless(name):
    return f"""
MODULE {name}(ins_valid, outs_ready)
    VAR 
    full : boolean;

    ASSIGN
    init(full) := FALSE;
    next(full) := (full <-> outs_ready) ? ins_valid : full;

    -- outputs
    DEFINE
    outs_valid := full | ins_valid;
    ins_ready := (!full) | outs_ready;
"""


def _generate_one_slot_break_none(name, data_type):
    return f"""
MODULE {name}(ins, ins_valid, outs_ready)
    VAR 
    reg : {data_type};
    full : boolean;

    ASSIGN
    init(reg) := {data_type.format_constant(0)};
    next(reg) := enable ? ins : reg;
    init(full) := FALSE;
    next(full) := (full <-> outs_ready) ? ins_valid : full;

    DEFINE
    enable := ins_valid & (outs_ready <-> full);
    outs := full ? reg : ins;
    outs_valid := full | ins_valid;
    ins_ready := (!full) | outs_ready;
"""


def _generate_slot_based_elastic_fifo_inner_dataless(name, slots):
    # fifo generated as chain of fully transparent slots for faster model checking
    slots_valid = ["ins_valid"] + \
        [f"b{i}.outs_valid" for i in range(slots - 1)]
    slots_ready = [
        f"b{i + 1}.ins_ready" for i in range(slots - 1)] + ["outs_ready"]
    return f"""
MODULE {name}(ins_valid, outs_ready)
    {"\n    ".join([f"VAR b{n} : {name}_tslot({valid}, {ready});" for n, (valid, ready) in enumerate(zip(slots_valid, slots_ready))])}

    -- outputs
		DEFINE outs_valid := b{slots - 1}.outs_valid;
		DEFINE ins_ready := b0.ins_ready;
{_generate_one_slot_break_none_dataless(f"{name}_tslot")}
"""


def _generate_elastic_fifo_inner_dataless_oneslot(name):
    return f"""
MODULE {name}(ins_valid, outs_ready)
  VAR
  full : boolean;
  empty : boolean;

  DEFINE
  read_en := outs_ready & !empty;
  write_en := ins_valid & (!full | outs_ready);

  init(full) := FALSE;
  next(full) := case
    write_en & !read_en : TRUE;
    !write_en & read_en : FALSE;
    TRUE : full;
  esac;

  init(empty) := TRUE;
  next(empty) := case
    !write_en & read_en : TRUE;
    write_en & !read_en : FALSE;
    TRUE : empty;
  esac;

  -- output
  DEFINE
  ins_ready := !full | outs_ready;
  outs_valid := !empty;
"""


def _generate_elastic_fifo_inner_dataless(name, slots):
    return f"""
MODULE {name}(ins_valid, outs_ready)
  VAR
  full : boolean;
  empty : boolean;
  head : 0..{slots - 1};
  tail : 0..{slots - 1};

  DEFINE
  read_en := outs_ready & !empty;
  write_en := ins_valid & (!full | outs_ready);

  ASSIGN
  init(tail) := 0;
  next(tail) := case
    {"\n    ".join([f"write_en & (tail = {n}) : {(n + 1) % slots};" for n in range(slots)])}
    TRUE : tail;
  esac;

  init(head) := 0;
  next(head) := case
    {"\n    ".join([f"read_en & (head = {n}) : {(n + 1) % slots};" for n in range(slots)])}
    TRUE : head;
  esac;

  init(full) := FALSE;
  next(full) := case
    write_en & !read_en : case
      tail < {slots - 1} & head = tail + 1: TRUE;
      tail = {slots - 1} & head = 0 : TRUE;
      TRUE : full;
    esac;
    !write_en & read_en : FALSE;
    TRUE : full;
  esac;

  init(empty) := TRUE;
  next(empty) := case
    !write_en & read_en : case
      head < {slots - 1} & tail = head + 1: TRUE;
      head = {slots - 1} & tail = 0 : TRUE;
      TRUE : empty;
    esac;
    write_en & !read_en : FALSE;
    TRUE : empty;
  esac;

  -- output
  DEFINE
  ins_ready := !full | outs_ready;
  outs_valid := !empty;
"""


def _generate_slot_based_elastic_fifo_inner(name, slots, data_type):
    # fifo generated as chain of fully transparent slots for faster model checking
    slots_data = ["ins"] + [f"b{i}.outs" for i in range(slots)]
    slots_valid = ["ins_valid"] + \
        [f"b{i}.outs_valid" for i in range(slots - 1)]
    slots_ready = [
        f"b{i + 1}.ins_ready" for i in range(slots - 1)] + ["outs_ready"]
    return f"""
MODULE {name}(ins, ins_valid, outs_ready)
    {"\n    ".join([f"VAR b{n} : {name}_tslot({data}, {valid}, {ready});" for n, (data, valid, ready) in enumerate(zip(slots_data, slots_valid, slots_ready))])}

    -- output
		DEFINE outs   := b{slots - 1}.outs;
		DEFINE outs_valid   := b{slots - 1}.outs_valid;
		DEFINE ins_ready   := b0.ins_ready;
{_generate_one_slot_break_none(f"{name}_tslot", data_type)}
"""


def _generate_elastic_fifo_inner(name, slots, data_type):
    return f"""
MODULE {name}(ins, ins_valid, outs_ready)
  {"\n  ".join([f"VAR mem_{n} : {data_type};" for n in range(slots)])}
  VAR
  full : boolean;
  empty : boolean;
  head : 0..{slots - 1};
  tail : 0..{slots - 1};

  DEFINE
  read_en := outs_ready & !empty;
  write_en := ins_valid & (!full | outs_ready);

  ASSIGN
  init(tail) := 0;
  next(tail) := case
    {"\n    ".join([f"write_en & (tail = {n}) : {(n + 1) % slots};" for n in range(slots)])}
    TRUE : tail;
  esac;

  init(head) := 0;
  next(head) := case
    {"\n    ".join([f"read_en & (head = {n}) : {(n + 1) % slots};" for n in range(slots)])}
    TRUE : head;
  esac;

  {"\n  ".join([f"""ASSIGN
  init(mem_{n}) := {data_type.format_constant(0)};
  next(mem_{n}) := write_en & (tail = {n}) ? ins : mem_{n};""" for n in range(slots)])}

  init(full) := FALSE;
  next(full) := case
    write_en & !read_en : case
      tail < {slots - 1} & head = tail + 1: TRUE;
      tail = {slots - 1} & head = 0 : TRUE;
      TRUE : full;
    esac;
    !write_en & read_en : FALSE;
    TRUE : full;
  esac;

  init(empty) := TRUE;
  next(empty) := case
    !write_en & read_en : case
      head < {slots - 1} & tail = head + 1: TRUE;
      head = {slots - 1} & tail = 0 : TRUE;
      TRUE : empty;
    esac;
    write_en & !read_en : FALSE;
    TRUE : empty;
  esac;

  -- output
  DEFINE
  ins_ready := !full | outs_ready;
  outs_valid := !empty;
  outs := case
    {"\n    ".join([f"head = {n} : mem_{n};" for n in range(slots)])}
    TRUE : mem_0;
  esac;
"""
