from generators.support.arith2 import generate_arith2


def generate_andi(name, params):
    bitwidth = params["bitwidth"]

    body = """
  result <= lhs and rhs;
    """

    return generate_arith2(
        name=name,
        modType="andi",
        bitwidth=bitwidth,
        body=body,
        extra_signals=params.get("extra_signals", None),
    )
