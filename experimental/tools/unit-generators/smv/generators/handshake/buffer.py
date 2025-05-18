from generators.support.utils import *
from generators.support.tfifo import generate_tfifo
from generators.support.tehb import generate_tehb
from generators.support.ofifo import generate_ofifo
from generators.support.oehb import generate_oehb


def generate_buffer(name, params):
  match_r = re.search(r"R:(\d+)", params[ATTR_TIMING].replace(" ", ""))
  timing_r = False if not match_r else bool(int(match_r.group(1)))
  match_d = re.search(r"D:(\d+)", params[ATTR_TIMING].replace(" ", ""))
  timing_d = False if not match_d else bool(int(match_d.group(1)))
  match_v = re.search(r"V:(\d+)", params[ATTR_TIMING].replace(" ", ""))
  timing_v = False if not match_v else bool(int(match_v.group(1)))
  transparent = timing_r
  slots = params[ATTR_SLOTS] if ATTR_SLOTS in params else 1
  mlir_data_type = params[ATTR_PORT_TYPES]["outs"]

  if transparent and slots > 1:
    return generate_tfifo(name, {ATTR_SLOTS: slots, ATTR_DATA_TYPE: mlir_data_type})
  elif transparent and slots == 1:
    return generate_tehb(name, {ATTR_DATA_TYPE: mlir_data_type})
  elif not transparent and slots > 1:
    return generate_ofifo(name, {ATTR_SLOTS: slots, ATTR_DATA_TYPE: mlir_data_type})
  elif not transparent and slots == 1:
    return generate_oehb(name, {ATTR_DATA_TYPE: mlir_data_type})
  else:
    raise ValueError(f"Buffer implementation not found")
