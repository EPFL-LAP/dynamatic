import ast

from generators.support.utils import VhdlScalarType
from generators.handshake.tfifo import generate_tfifo
from generators.handshake.tehb import generate_tehb
from generators.handshake.ofifo import generate_ofifo
from generators.handshake.oehb import generate_oehb

def generate_buffer(name, params):
  timing = ast.literal_eval(params["timing"])
  num_slots = int(params["num_slots"])

  timing_r = bool(timing["ready_latency"])
  timing_d = bool(timing["data_latency"])
  timing_v = bool(timing["valid_latency"])
  transparent = timing_r and not (timing_d or timing_v)

  if transparent and num_slots > 1:
    return generate_tfifo(name, params)
  elif transparent and num_slots == 1:
    return generate_tehb(name, params)
  elif not transparent and num_slots > 1:
    return generate_ofifo(name, params)
  elif not transparent and num_slots == 1:
    return generate_oehb(name, params)
  raise ValueError("Invalid buffer configuration")
