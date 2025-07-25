from generators.support.arith2 import generate_arith2


def generate_shli(name, params):
    bitwidth = params["bitwidth"]

    body = f"""
  result <= std_logic_vector(shift_left(unsigned(lhs), to_integer(unsigned('0' & rhs({bitwidth} - 2 downto 0)))));
    """

    return generate_arith2(
        name=name,
        modType="shli",
        bitwidth=bitwidth,
        body=body,
        extra_signals=params.get("extra_signals", None),
    )
