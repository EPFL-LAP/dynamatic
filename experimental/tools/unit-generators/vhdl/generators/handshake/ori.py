from generators.support.arith2 import generate_arith2


def generate_ori(name, params):
    bitwidth = params["bitwidth"]

    body = """
  result <= lhs or rhs;
    """

    return generate_arith2(
        name=name,
        modType="ori",
        bitwidth=bitwidth,
        body=body,
        extra_signals=params.get("extra_signals", None),
    )
