from generators.support.arith2 import generate_arith_binary


def generate_shrsi(name, params):
    bitwidth = params["bitwidth"]

    body = f"""
  result <= std_logic_vector(shift_right(signed(lhs), to_integer(signed('0' & rhs({bitwidth} - 2 downto 0)))));
    """

    return generate_arith_binary(
        name=name,
        op_type="shrsi",
        bitwidth=bitwidth,
        body=body,
        extra_signals=params.get("extra_signals", None)
    )
