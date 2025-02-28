import re

from generators.handshake.tfifo import generate_tfifo
from generators.handshake.tehb import generate_tehb
from generators.handshake.ofifo import generate_ofifo
from generators.handshake.oehb import generate_oehb


def generate_buffer(name, params):
  num_slots = params["num_slots"]

  match_r = re.search(r"R: (\d+)", params["timing"])
  timing_r = False if not match_r else bool(match_r[0])
  match_d = re.search(r"D: (\d+)", params["timing"])
  timing_d = False if not match_d else bool(match_d[0])
  match_v = re.search(r"V: (\d+)", params["timing"])
  timing_v = False if not match_v else bool(match_v[0])
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
