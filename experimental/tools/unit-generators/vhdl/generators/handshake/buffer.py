from generators.handshake.tfifo import generate_tfifo
from generators.handshake.tehb import generate_tehb
from generators.handshake.fifo_break_dv import generate_fifo_break_dv
from generators.handshake.one_slot_break_dvr import generate_one_slot_break_dvr
from generators.handshake.shift_reg_break_dvr import generate_shift_reg_break_dv
from generators.handshake.oehb import generate_oehb

from support.utils import try_enum_cast

from enum import Enum

class BufferType(Enum):
    ONE_SLOT_BREAK_DV = "ONE_SLOT_BREAK_DV"
    ONE_SLOT_BREAK_R = "ONE_SLOT_BREAK_R"
    FIFO_BREAK_NONE = "FIFO_BREAK_NONE"
    FIFO_BREAK_DV = "FIFO_BREAK_DV"
    ONE_SLOT_BREAK_DVR = "ONE_SLOT_BREAK_DVR"
    SHIFT_REG_BREAK_DV = "SHIFT_REG_BREAK_DV"


def generate_buffer(name, params):

    buffer_type = try_enum_cast(params["buffer_type"], BufferType)

    match buffer_type:
        case BufferType.ONE_SLOT_BREAK_R:
            return generate_tehb(name, params)
        case BufferType.FIFO_BREAK_NONE:
            return generate_tfifo(name, params)
        case BufferType.ONE_SLOT_BREAK_DV:
            return generate_oehb(name, params)
        case BufferType.FIFO_BREAK_DV:
            return generate_fifo_break_dv(name, params)
        case BufferType.ONE_SLOT_BREAK_DVR:
            return generate_one_slot_break_dvr(name, params)
        case BufferType.SHIFT_REG_BREAK_DV:
            return generate_shift_reg_break_dv(name, params)
        case _:
            raise ValueError(f"Unhandled buffer type: {buffer_type}")
