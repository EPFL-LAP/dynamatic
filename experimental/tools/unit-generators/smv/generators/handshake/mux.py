from generators.handshake.buffer import generate_buffer


def generate_mux(name, params):
  if "data_type" not in params or params["data_type"] == "!handshake.control<>":
    return _generate_mux_dataless(name, params["size"])
  else:
    return _generate_mux(name, params["size"], params["data_type"])


def _generate_mux_dataless(name, size):
  return f"""
MODULE {name}({", ".join([f"ins_valid_{n}" for n in range(size)])}, index, index_valid, outs_ready)
    VAR inner_tehb : {name}__tehb(tehb_ins_valid, outs_ready);

    DEFINE tehb_ins_valid := case
                               {"\n                               ".join([f"index = {n} : index_valid & ins_valid_{n};" for n in range(size)])}
                               TRUE : FALSE;
                             esac;

    // output
    {"\n    ".join([f"DEFINE ins_ready_{n} := index = {n} & index_valid & tehb_inner.ins_ready & ins_valid_{n} | !ins_valid{n};" for n in range(size)])}
    DEFINE index_ready := !index_valid | tehb_ins_valid & tehb_inner.ins_ready;
    DEFINE outs_valid := tehb_inner.outs_valid;

{generate_buffer(f"{name}__tehb_dataless", {"slots": 1, "timing": "R: 1"})}
"""


def _generate_mux(name, size, data_type):
  return f"""
MODULE {name}({", ".join([f"ins_{n}" for n in range(size)])}, {", ".join([f"ins_valid_{n}" for n in range(size)])}, index, index_valid, outs_ready)
    VAR inner_tehb : {name}__tehb(tehb_ins, tehb_ins_valid, outs_ready);

    DEFINE tehb_ins := case
                         {"\n                         ".join([f"index = {n} & index_valid & ins_valid_{n} : ins_{n};" for n in range(size)])}
                         TRUE : ins_0;
                       esac;
    DEFINE tehb_ins_valid := case
                               {"\n                               ".join([f"index = {n} : index_valid & ins_valid_{n} | !ins_valid{n};" for n in range(size)])}
                               TRUE : FALSE;
                             esac;

    // output
    {"\n    ".join([f"DEFINE ins_ready_{n} := index = {n} & index_valid & tehb_inner.ins_ready & ins_valid_{n} | !ins_valid{n};" for n in range(size)])}
    DEFINE index_ready := !index_valid | tehb_ins_valid & tehb_inner.ins_ready;
    DEFINE outs_valid := tehb_inner.outs_valid;
    DEFINE outs := tehb_inner.outs;

{generate_buffer(f"{name}__tehb", {"slots": 1, "timing": "R: 1", "data_type": data_type})}
"""


if __name__ == "__main__":
  print(generate_mux("test_mux_dataless", {"size": 4}))
  print(generate_mux("test_mux", {"size": 2, "data_type": "!handshake.channel<i32>"}))
