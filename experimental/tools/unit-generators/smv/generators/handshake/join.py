def generate_join(name, params):
    return _generate_join(name, params["size"])


def _generate_join(name, size):
    return f"""
MODULE {name}({", ".join([f"ins_valid_{n}" for n in range(size)])}, outs_ready)

    DEFINE all_valid = {" & ".join([f"ins_valid_{n}" for n in range(size)])};

    // output
    {"\n    ".join([f"DEFINE ins_ready_{n} := outs_ready & {" & ".join([f"ins_valid_{m}" for m in range(size) if m != n])}" for n in range(size)])}
    DEFINE outs_valid := all_valid;
"""


if __name__ == "__main__":
    print(generate_join("test_fork_dataless", {"size": 3}))
