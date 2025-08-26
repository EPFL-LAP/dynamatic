from generators.support.arith_binary import generate_arith_binary


def generate_xori(name, params):
    bitwidth = params["bitwidth"]

    body = """
  result <= lhs xor rhs;
    """

    return generate_arith_binary(
        name=name,
        handshake_op="xori",
        bitwidth=bitwidth,
        body=body,
        extra_signals=params.get("extra_signals", None),
    )
