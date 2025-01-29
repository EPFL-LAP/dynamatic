from generators.support.elastic_fifo_inner import generate_elastic_fifo_inner
from generators.support.utils import *


def generate_buffer(name, params):
  timing_r = bool(re.search(r"R: (\d+)", params["timing"]))
  timing_d = bool(re.search(r"D: (\d+)", params["timing"]))
  timing_v = bool(re.search(r"V: (\d+)", params["timing"]))
  transparent = timing_r and not (timing_d or timing_v)
  slots = params["slots"] if "slots" in params else 1
  data_type = None if "data_type" not in params or params["data_type"] == "!handshake.control<>" else mlir_type_to_smv_type(
      params["data_type"])

  if transparent and slots > 1 and data_type is None:
    return _generate_tfifo_dataless(name, slots)
  elif transparent and slots > 1 and data_type is not None:
    return _generate_tfifo(name, slots, data_type)
  elif transparent and slots == 1 and data_type is None:
    return _generate_tehb_dataless(name)
  elif transparent and slots == 1 and data_type is not None:
    return _generate_tehb(name, data_type)
  elif not transparent and slots > 1 and data_type is None:
    return _generate_ofifo_dataless(name, slots)
  elif not transparent and slots > 1 and data_type is not None:
    return _generate_ofifo(name, slots, data_type)
  elif not transparent and slots == 1 and data_type is None:
    return _generate_oehb_dataless(name)
  elif not transparent and slots == 1 and data_type is not None:
    return _generate_oehb(name, data_type)


def _generate_oehb_dataless(name):
  return f"""
MODULE {name} (ins_valid, outs_ready)
    VAR outs_valid_i : boolean;

    ASSIGN
    init(outs_valid_i) := FALSE;
    next(outs_valid_i) := ins_valid | (outs_valid_i & !outs_ready);

    // output
    DEFINE ins_ready := !outs_valid_i | outs_ready;
    DEFINE outs_valid := outs_valid_i;
"""


def _generate_oehb(name, data_type):
  return f"""
MODULE {name} (ins, ins_valid, outs_ready)
    VAR inner_oehb : {name}__oehb_dataless(ins_valid, outs_ready);
    VAR outs_i : {data_type};

    ASSIGN
    init(outs_i) := {smv_init_data_type(data_type)};
    next(outs_i) := case
                      ins_ready & ins_valid : ins;
                      TRUE : outs_i;
                    esac;
    
    // output
    DEFINE ins_ready := inner_oehb.ins_ready;
    DEFINE outs_valid := inner_oehb.outs_valid;
    DEFINE outs := outs_i;

{_generate_oehb_dataless(f"{name}__oehb_dataless")}
"""


def _generate_ofifo_dataless(name, slots):
  return f"""
MODULE {name} (ins_valid, outs_ready)
    VAR inner_tehb : {name}__tehb_dataless(ins_valid, inner_elastic_fifo.ins_ready);
    VAR inner_elastic_fifo : {name}__elastic_fifo_inner_dataless(inner_tehb.outs_valid, outs_ready);

    // output
    DEFINE ins_ready := inner_tehb.ins_ready;
    DEFINE outs_valid := inner_elastic_fifo.outs_valid;

{_generate_tehb_dataless(f"{name}__tehb_dataless")}
{generate_elastic_fifo_inner(f"{name}__elastic_fifo_inner_dataless", slots)}
"""


def _generate_ofifo(name, slots, data_type):
  return f"""
MODULE {name} (ins, ins_valid, outs_ready)
    VAR inner_tehb : {name}__tehb(ins, ins_valid, inner_elastic_fifo.ins_ready);
    VAR inner_elastic_fifo : {name}__elastic_fifo_inner(inner_tehb.outs, inner_tehb.outs_valid, outs_ready);

    // output
    DEFINE ins_ready := inner_tehb.ins_ready;
    DEFINE outs_valid := inner_elastic_fifo.outs_valid;
    DEFINE outs := inner_elastic_fifo.outs;

{_generate_tehb(f"{name}__tehb_dataless", data_type)}
{generate_elastic_fifo_inner(f"{name}__elastic_fifo_inner_dataless", slots, data_type)}
"""


def _generate_tehb_dataless(name):
  return f"""
MODULE {name}(ins_valid, outs_ready)
    VAR full : boolean;

    ASSIGN
    init(full) := FALSE;
    next(full) := outs_valid & !outs_ready;

    // output
    DEFINE ins_ready := !full;
    DEFINE outs_valid := ins_valid | full;
"""


def _generate_tehb(name, data_type):
  return f"""
MODULE {name}(ins, ins_valid, outs_ready)
    VAR inner_tehb : {name}__tehb_dataless(ins_valid, outs_ready);
    VAR data : {data_type};

    ASSIGN
    init(outs_i) := {smv_init_data_type(data_type)};
    next(data) := ins_ready & ins_valid & !outs_ready ? ins : data;

    // output
    DEFINE ins_ready := inner_tehb.ins_ready;
    DEFINE outs_valid := inner_tehb.outs_valid;
    DEFINE outs := tehb_dataless.full ? data : ins;
{_generate_tehb_dataless(f"{name}__tehb_dataless")}
"""


def _generate_tfifo_dataless(name, slots):
  return f"""
MODULE {name} (ins_valid, outs_ready)
    VAR inner_elastic_fifo : {name}__elastic_fifo_inner_dataless(fifo_valid, fifo_ready);

    DEFINE fifo_valid := ins_valid & (!outs_ready | inner_elastic_fifo.outs_valid);
    DEFINE fifo_ready := outs_ready;

    // output
    DEFINE ins_ready := inner_elastic_fifo.ins_ready | outs_ready;
    DEFINE outs_valid := ins_valid | inner_elastic_fifo.outs_valid;

{generate_elastic_fifo_inner(f"{name}__elastic_fifo_inner_dataless", slots)}
"""


def _generate_tfifo(name, slots, data_type):
  return f"""
MODULE {name} (ins, ins_valid, outs_ready)
    VAR inner_elastic_fifo : {name}__elastic_fifo_inner(fifo_valid, fifo_ready);

    DEFINE fifo_valid := ins_valid & (!outs_ready | inner_elastic_fifo.outs_valid);
    DEFINE fifo_ready := outs_ready;

    // output
    DEFINE ins_ready := inner_elastic_fifo.ins_ready | outs_ready;
    DEFINE outs_valid := ins_valid | inner_elastic_fifo.outs_valid;
    DEFINE outs := inner_elastic_fifo.outs_valid ? inner_elastic_fifo.outs : ins;

{generate_elastic_fifo_inner(f"{name}__elastic_fifo_inner", slots, data_type)}
"""


if __name__ == "__main__":
  print(
      generate_buffer(
          "test_tfifo_dataless", {
              "timing": "#handshake<timing {{R: 1}}", "slots": 5}
      )
  )
  print(
      generate_buffer(
          "test_tfifo",
          {"timing": "#handshake<timing {{R: 1}}",
              "slots": 5, "data_type": "!handshake.channel<i1>"},
      )
  )
  print(
      generate_buffer("test_tehb_dataless", {
                      "timing": "#handshake<timing {{R: 1}}"})
  )
  print(
      generate_buffer(
          "test_tehb", {
              "timing": "#handshake<timing {{R: 1}}", "data_type": "!handshake.channel<i32>"}
      )
  )
  print(
      generate_buffer(
          "test_ofifo_dataless", {
              "timing": "#handshake<timing {{R: 0}}", "slots": 5}
      )
  )
  print(
      generate_buffer(
          "test_ofifo",
          {"timing": "#handshake<timing {{R: 0}}",
              "slots": 5, "data_type": "!handshake.channel<i1>"},
      )
  )
  print(
      generate_buffer("test_oehb_dataless", {
                      "timing": "#handshake<timing {{R: 0}}"})
  )
  print(
      generate_buffer(
          "test_oehb", {
              "timing": "#handshake<timing {{R: 0}}", "data_type": "!handshake.channel<i32>"}
      )
  )
