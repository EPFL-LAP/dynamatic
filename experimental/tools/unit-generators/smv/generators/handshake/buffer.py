from generators.support.utils import *
from generators.support.tfifo import generate_tfifo
from generators.support.tehb import generate_tehb
from generators.support.ofifo import generate_ofifo
from generators.support.oehb import generate_oehb


def generate_buffer(name, params):
    slots = params[ATTR_SLOTS]
    transparent = params[ATTR_TRANSPARENT]
    bitwidth = params[ATTR_BITWIDTH]

    if transparent and slots > 1:
        return generate_tfifo(name, {ATTR_SLOTS: slots, ATTR_BITWIDTH: bitwidth})
    elif transparent and slots == 1:
        return generate_tehb(name, {ATTR_BITWIDTH: bitwidth})
    elif not transparent and slots > 1:
        return generate_ofifo(name, {ATTR_SLOTS: slots, ATTR_BITWIDTH: bitwidth})
    elif not transparent and slots == 1:
        return generate_oehb(name, {ATTR_BITWIDTH: bitwidth})
    else:
        raise ValueError(f"Buffer implementation not found")
