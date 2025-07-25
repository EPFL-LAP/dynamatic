from generators.support.arith1 import generate_arith1


def generate_logical_not(name, params):
    bitwidth = params["bitwidth"]

    modType = "not"

    body = f"""
  outs       <= not ins;
    """

    return generate_arith1(
        name=name,
        modType=modType,
        bitwidth=bitwidth,
        body=body,
        extra_signals=params.get("extra_signals", None),
    )
