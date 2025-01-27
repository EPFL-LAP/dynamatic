def generate_lazy_fork(name, params):
  if "data_type" not in params or params["data_type"] == "!handshake.control<>":
    return _generate_lazy_fork_dataless(name, params["size"])
  else:
    return _generate_lazy_fork(name, params["size"], params["data_type"])


def _generate_lazy_fork_dataless(name, size):
  return f"""
MODULE {name}(ins_valid, {", ".join([f"outs_ready_{n}" for n in range(size)])})

    DEFINE all_ready = {" & ".join([f"outs_ready_{n}" for n in range(size)])};

    // output
    DEFINE ins_ready := all_ready;
    {"\n    ".join([f"DEFINE outs_valid_{n} := ins_valid & all_ready;" for n in range(size)])}
"""


def _generate_lazy_fork(name, size, data_type):
  return f"""
MODULE {name}(ins, ins_valid, {", ".join([f"outs_ready_{n}" for n in range(size)])})
    VAR inner_lazy_fork : {name}__lazy_fork_dataless(ins_valid, {", ".join([f"outs_ready_{n}" for n in range(size)])});

    //output
    DEFINE ins_ready = inner_lazy_fork.ins_ready;
    {"\n    ".join([f"DEFINE outs_valid_{n} := inner_lazy_fork.outs_valid_{n};" for n in range(size)])}
    {"\n    ".join([f"DEFINE outs_{n} := ins;" for n in range(size)])}

{_generate_lazy_fork_dataless(f"{name}__lazy_fork_dataless", size)}
"""


if __name__ == "__main__":
  print(generate_lazy_fork("test_lazy_fork_dataless", {"size": 4}))
  print(generate_lazy_fork("test_lazy_fork", {"size": 2, "data_type": "int"}))
