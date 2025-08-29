from generators.support.utils import *
from generators.support.buffer_counter import generate_buffer_counter


def generate_fifo_break_none(name, params):
    slots = params[ATTR_SLOTS]
    data_type = SmvScalarType(params[ATTR_BITWIDTH])
    debug_counter = params.get(ATTR_DEBUG_COUNTER, False)

    if data_type.bitwidth == 0:
        if slots == 1:
            return _generate_one_slot_break_none_dataless(name, debug_counter)
        else:
            return _generate_fifo_break_none_dataless(name, slots, debug_counter)
    else:
        if slots == 1:
            return _generate_one_slot_break_none(name, data_type, debug_counter)
        else:
            return _generate_fifo_break_none(name, slots, data_type, debug_counter)


def _generate_one_slot_break_none_dataless(name, debug_counter):
    return f"""
MODULE {name}(ins_valid, outs_ready)
    VAR 
    full : boolean;
    {f"debug_counter : {name}__debug_counter(ins_valid, ins_ready, outs_valid, outs_ready);" if debug_counter else ""}

    ASSIGN
    init(full) := FALSE;
    next(full) := (full <-> outs_ready) ? ins_valid : full;

    -- outputs
    DEFINE
    outs_valid := full | ins_valid;
    ins_ready := (!full) | outs_ready;
{generate_buffer_counter(f"{name}__debug_counter", 1) if debug_counter else ""}
"""


def _generate_one_slot_break_none(name, data_type, debug_counter):
    return f"""
MODULE {name}(ins, ins_valid, outs_ready)
    VAR 
    reg : {data_type};
    full : boolean;
    {f"debug_counter : {name}__debug_counter(ins_valid, ins_ready, outs_valid, outs_ready);" if debug_counter else ""}

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
{generate_buffer_counter(f"{name}__debug_counter", 1) if debug_counter else ""}
"""


def _generate_fifo_break_none_dataless(name, slots, debug_counter):
    # fifo generated as chain of fully transparent slots for faster model checking
    slots_valid = ["ins_valid"] + \
        [f"b{i}.outs_valid" for i in range(slots - 1)]
    slots_ready = [
        f"b{i + 1}.ins_ready" for i in range(slots - 1)] + ["outs_ready"]
    return f"""
MODULE {name}(ins_valid, outs_ready)
    {"\n    ".join([f"VAR b{n} : {name}_tslot({valid}, {ready});" for n, (valid, ready) in enumerate(zip(slots_valid, slots_ready))])}
    {f"VAR debug_counter : {name}__debug_counter(ins_valid, ins_ready, outs_valid, outs_ready);" if debug_counter else ""}

    -- outputs
		DEFINE outs_valid := b{slots - 1}.outs_valid;
		DEFINE ins_ready := b0.ins_ready;
{_generate_one_slot_break_none_dataless(f"{name}_tslot", False)}
{generate_buffer_counter(f"{name}__debug_counter", slots) if debug_counter else ""}
"""


def _generate_fifo_break_none(name, slots, data_type, debug_counter):
    # fifo generated as chain of fully transparent slots for faster model checking
    slots_data = ["ins"] + [f"b{i}.outs" for i in range(slots)]
    slots_valid = ["ins_valid"] + \
        [f"b{i}.outs_valid" for i in range(slots - 1)]
    slots_ready = [
        f"b{i + 1}.ins_ready" for i in range(slots - 1)] + ["outs_ready"]
    return f"""
MODULE {name}(ins, ins_valid, outs_ready)
    {"\n    ".join([f"VAR b{n} : {name}_tslot({data}, {valid}, {ready});" for n, (data, valid, ready) in enumerate(zip(slots_data, slots_valid, slots_ready))])}
    {f"VAR debug_counter : {name}__debug_counter(ins_valid, ins_ready, outs_valid, outs_ready);" if debug_counter else ""}

    -- output
		DEFINE outs   := b{slots - 1}.outs;
		DEFINE outs_valid   := b{slots - 1}.outs_valid;
		DEFINE ins_ready   := b0.ins_ready;
{_generate_one_slot_break_none(f"{name}_tslot", data_type, False)}
{generate_buffer_counter(f"{name}__debug_counter", slots) if debug_counter else ""}
"""
