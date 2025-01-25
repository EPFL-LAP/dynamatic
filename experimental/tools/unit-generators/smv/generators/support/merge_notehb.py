def generate_merge_notehb(name, size, data_type=None):
    if data_type is None:
        return _generate_merge_notehb_dataless(name, size)
    else:
        return _generate_merge_notehb(name, size, data_type)


def _generate_merge_notehb_dataless(name, size):
    return f"""
MODULE {name}({", ".join([f"ins_valid_{n}" for n in range(size)])}, outs_ready)

    DEFINE one_valid := {' | '.join([f'ins_valid_{i}' for i in range(size)])};

    // output
    {"\n    ".join([f"DEFINE ins_ready_{n} := ins_valid_{n} & outs_ready;" for n in range(size)])}
    DEFINE outs_valid := one_valid;
"""


def _generate_merge_notehb(name, size, data_type):
    return f"""
MODULE {name}({", ".join([f"ins_{n}" for n in range(size)])}, {", ".join([f"ins_valid_{n}" for n in range(size)])}, outs_ready)

    DEFINE one_valid := {' | '.join([f'ins_valid_{i}' for i in range(size)])};
    DEFINE data := case
                     {"\n                     ".join([f"ins_valid_{n} : ins_{n};" for n in range(size)])}
                     TRUE : FALSE;
                   esac;

    // output
    {"\n    ".join([f"DEFINE ins_ready_{n} := ins_valid_{n} & outs_ready;" for n in range(size)])}
    DEFINE outs_valid := one_valid;
    DEFINE outs := data;
"""