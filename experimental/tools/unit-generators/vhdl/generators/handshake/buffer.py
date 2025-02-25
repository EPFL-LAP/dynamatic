import re

from generators.support.tfifo import generate_tfifo
from generators.support.tehb import generate_tehb
from generators.support.ofifo import generate_ofifo
from generators.support.oehb import generate_oehb


def generate_buffer(name, params):
  num_slots = params["num_slots"]

  timing_r = bool(re.search(r"R: (\d+)", params["timing"]))
  timing_d = bool(re.search(r"D: (\d+)", params["timing"]))
  timing_v = bool(re.search(r"V: (\d+)", params["timing"]))
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
