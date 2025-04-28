from generators.handshake.tfifo import generate_tfifo
from generators.handshake.tehb import generate_tehb
from generators.handshake.ofifo import generate_ofifo
from generators.handshake.oehb import generate_oehb


def generate_buffer(name, params):
    num_slots = params["num_slots"]
    transparent = params["transparent"]

    if transparent and num_slots > 1:
        return generate_tfifo(name, params)
    elif transparent and num_slots == 1:
        return generate_tehb(name, params)
    elif not transparent and num_slots > 1:
        return generate_ofifo(name, params)
    elif not transparent and num_slots == 1:
        return generate_oehb(name, params)
    raise ValueError("Invalid buffer configuration")
