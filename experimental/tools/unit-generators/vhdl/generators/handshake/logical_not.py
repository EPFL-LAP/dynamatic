from generators.support.arith1 import generate_arith1


def generate_logical_not(name, params):
    bitwidth = params["bitwidth"]

    body = f"""
  outs       <= not ins;
    """

    return generate_arith1(
        name=name,
        modType="not",
        bitwidth=bitwidth,
        body=body,
        extra_signals=params.get("extra_signals", None),
    )
