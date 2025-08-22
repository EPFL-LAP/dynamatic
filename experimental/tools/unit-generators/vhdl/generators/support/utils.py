from typing import Type, Union
from enum import Enum
from generators.handshake.buffers.one_slot_break_dv import generate_one_slot_break_dv
from generators.handshake.buffers.shift_reg_break_dv import generate_shift_reg_break_dv

def data(code: str, bitwidth: int) -> str:
    return code if bitwidth else ""


# Define the type for extra signals, which are stored as a dictionary with signal names and their bitwidths.
ExtraSignals = dict[str, int]


def try_enum_cast(value: str, enum_class: Type[Enum]) -> Union[Enum, str]:
    try:
        return enum_class(value)
    except ValueError:
        return value

def generate_valid_propagation_buffer(name, latency):
    if latency == 1:
        return generate_one_slot_break_dv(name, {"bitwidth": 0})
    else:
        return generate_shift_reg_break_dv(name, {"bitwidth": 0, "num_slots": latency})


VIVADO_IMPL = "vivado"
FLOPOCO_IMPL = "flopoco"
