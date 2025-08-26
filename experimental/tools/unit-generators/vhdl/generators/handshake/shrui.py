from generators.support.arith2 import generate_arith_binary


def generate_shrui(name, params):
    bitwidth = params["bitwidth"]

    body = f"""
  result <= std_logic_vector(shift_right(unsigned(lhs), to_integer(unsigned('0' & rhs(DATA_TYPE - 2 downto 0)))));
    """

    return generate_arith_binary(
        name=name,
        op_type="shrui",
        bitwidth=bitwidth,
        body=body,
        extra_signals=params.get("extra_signals", None)
    )
