from generators.support.arith2 import generate_arith_binary


def generate_xori(name, params):
    bitwidth = params["bitwidth"]

    body = """
  result <= lhs xor rhs;
    """

    return generate_arith_binary(
        name=name,
        op_type="xori",
        bitwidth=bitwidth,
        body=body,
        extra_signals=params.get("extra_signals", None),
    )
