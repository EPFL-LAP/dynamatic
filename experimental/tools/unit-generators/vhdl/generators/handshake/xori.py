from generators.support.arith2 import generate_arith2


def generate_xori(name, params):
    bitwidth = params["bitwidth"]

    body = """
  result <= lhs xor rhs;
    """

    return generate_arith2(
        name=name,
        modType="xori",
        bitwidth=bitwidth,
        body=body,
        extra_signals=params.get("extra_signals", None),
    )
