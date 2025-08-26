from generators.support.arith2 import generate_arith_binary


def generate_andi(name, params):
    bitwidth = params["bitwidth"]

    body = """
  result <= lhs and rhs;
    """

    return generate_arith_binary(
        name=name,
        op_type="andi",
        bitwidth=bitwidth,
        body=body,
        extra_signals=params.get("extra_signals", None),
    )
