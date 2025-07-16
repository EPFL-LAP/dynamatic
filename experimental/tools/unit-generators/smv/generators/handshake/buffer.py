from generators.support.utils import *
from generators.support.tfifo import generate_tfifo
from generators.support.tehb import generate_tehb
from generators.support.ofifo import generate_ofifo
from generators.support.oehb import generate_oehb

from support.utils import try_enum_cast


class BufferType(Enum):
    ONE_SLOT_BREAK_DV = "ONE_SLOT_BREAK_DV"
    ONE_SLOT_BREAK_R = "ONE_SLOT_BREAK_R"
    FIFO_BREAK_NONE = "FIFO_BREAK_NONE"
    FIFO_BREAK_DV = "FIFO_BREAK_DV"
    ONE_SLOT_BREAK_DVR = "ONE_SLOT_BREAK_DVR"
    SHIFT_REG_BREAK_DV = "SHIFT_REG_BREAK_DV"


def generate_buffer(name, params):
    slots = params[ATTR_SLOTS]
    bitwidth = params[ATTR_BITWIDTH]

    buffer_type = try_enum_cast(params[ATTR_BUFFER_TYPE], BufferType)

    match buffer_type:
        case BufferType.ONE_SLOT_BREAK_R:
            return generate_tehb(name, {ATTR_BITWIDTH: bitwidth})
        case BufferType.FIFO_BREAK_NONE:
            return generate_tfifo(name, {ATTR_SLOTS: slots, ATTR_BITWIDTH: bitwidth})
        case BufferType.ONE_SLOT_BREAK_DV:
            return generate_oehb(name, params)
        case BufferType.FIFO_BREAK_DV:
            # this is not an ofifo
            # but it is what was being generated based on the previous code
            return generate_ofifo(name, {ATTR_SLOTS: slots, ATTR_BITWIDTH: bitwidth})
        case BufferType.ONE_SLOT_BREAK_DVR:
            # this is not an ofifo
            # but it is what was being generated based on the previous code
            return generate_ofifo(name, {ATTR_SLOTS: slots, ATTR_BITWIDTH: bitwidth})
        case BufferType.SHIFT_REG_BREAK_DV:
            # this is not an ofifo
            # but it is what was being generated based on the previous code
            return generate_ofifo(name, {ATTR_SLOTS: slots, ATTR_BITWIDTH: bitwidth})
        case _:
            raise ValueError(f"Unhandled buffer type: {buffer_type}")
