from generators.support.arith_binary import generate_arith_binary


def generate_subi(name, params):
    bitwidth = params["bitwidth"]

    body = """
  result <= std_logic_vector(unsigned(lhs) - unsigned(rhs));
    """

    return generate_arith_binary(
        name=name,
        handshake_op="subi",
        bitwidth=bitwidth,
        body=body,
        extra_signals=params.get("extra_signals", None)
    )
