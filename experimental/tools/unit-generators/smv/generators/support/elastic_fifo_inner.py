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
            return _generate_one_slot_elastic_fifo_inner_dataless(name)
        else:
            return _generate_one_slot_elastic_fifo_inner(name, data_type)


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


def _generate_one_slot_elastic_fifo_inner_dataless(name):
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


def _generate_one_slot_elastic_fifo_inner(name, data_type):
    return f"""
MODULE {name}(ins, ins_valid, outs_ready)
  VAR
  mem : {data_type};
  full : boolean;
  empty : boolean;

  DEFINE
  read_en := outs_ready & !empty;
  write_en := ins_valid & (!full | outs_ready);

  ASSIGN
  init(mem) := {data_type.format_constant(0)};
  next(mem) := write_en ? ins : mem;

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
  outs := mem;
"""
