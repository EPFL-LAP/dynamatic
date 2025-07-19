from generators.support.utils import *


def generate_fifo_break_none(name, params):
    slots = params[ATTR_SLOTS]
    data_type = SmvScalarType(params[ATTR_BITWIDTH])

    if data_type.bitwidth == 0:
        if slots == 1:
            return _generate_one_slot_break_none_dataless(name)
        else:
            return _generate_fifo_break_none_dataless(name, slots)
    else:
        if slots == 1:
            return _generate_one_slot_break_none(name, data_type)
        else:
            return _generate_fifo_break_none(name, slots, data_type)


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


def _generate_fifo_break_none_dataless(name, slots):
    # fifo generated as chain of fully transparent slots for faster model checking
    slots_valid = ["ins_valid"] + [f"b{i}.outs_valid" for i in range(slots - 1)]
    slots_ready = [f"b{i + 1}.ins_ready" for i in range(slots - 1)] + ["outs_ready"]
    return f"""
MODULE {name}(ins_valid, outs_ready)
    {"\n    ".join([f"VAR b{n} : {name}_tslot({valid}, {ready});" for n, (valid, ready) in enumerate(zip(slots_valid, slots_ready))])}

    -- outputs
		DEFINE outs_valid := b{slots - 1}.outs_valid;
		DEFINE ins_ready := b0.ins_ready;
{_generate_one_slot_break_none_dataless(f"{name}_tslot")}
"""


def _generate_fifo_break_none(name, slots, data_type):
    # fifo generated as chain of fully transparent slots for faster model checking
    slots_data = ["ins"] + [f"b{i}.outs" for i in range(slots)]
    slots_valid = ["ins_valid"] + [f"b{i}.outs_valid" for i in range(slots - 1)]
    slots_ready = [f"b{i + 1}.ins_ready" for i in range(slots - 1)] + ["outs_ready"]
    return f"""
MODULE {name}(ins, ins_valid, outs_ready)
    {"\n    ".join([f"VAR b{n} : {name}_tslot({data}, {valid}, {ready});" for n, (data, valid, ready) in enumerate(zip(slots_data, slots_valid, slots_ready))])}

    -- output
		DEFINE outs   := b{slots - 1}.outs;
		DEFINE outs_valid   := b{slots - 1}.outs_valid;
		DEFINE ins_ready   := b0.ins_ready;
{_generate_one_slot_break_none(f"{name}_tslot", data_type)}
"""
