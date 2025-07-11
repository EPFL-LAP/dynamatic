from generators.handshake.tfifo import generate_tfifo
from generators.handshake.tehb import generate_tehb
from generators.handshake.ofifo import generate_ofifo
from generators.handshake.oehb import generate_oehb

from enum import Enum

class BufferType(Enum):
    ONE_SLOT_BREAK_DV = "ONE_SLOT_BREAK_DV"
    ONE_SLOT_BREAK_R = "ONE_SLOT_BREAK_R"
    FIFO_BREAK_NONE = "FIFO_BREAK_NONE"
    FIFO_BREAK_DV = "FIFO_BREAK_DV"
    ONE_SLOT_BREAK_DVR = "ONE_SLOT_BREAK_DVR"
    SHIFT_REG_BREAK_DV = "SHIFT_REG_BREAK_DV"


def generate_buffer(name, params):
    num_slots = params["num_slots"]
    transparent = params["transparent"]

    try:
        print(params["buffer_type"])
        buffer_type = BufferType(params["buffer_type"])
    except ValueError:
        raise ValueError(f"Invalid buffer_type: '{params['buffer_type']}'. "
                         f"Beta backend supports: {[bt.value for bt in BufferType]}")

    print(buffer_type)

    if transparent and num_slots > 1:
        return generate_tfifo(name, params)
    elif transparent and num_slots == 1:
        return generate_tehb(name, params)
    elif not transparent and num_slots > 1:
        return generate_ofifo(name, params)
    elif not transparent and num_slots == 1:
        return generate_oehb(name, params)
    raise ValueError("Invalid buffer configuration")
