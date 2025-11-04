from generators.support.unary import generate_unary


def generate_logical_not(name, params):
    bitwidth = params["bitwidth"]

    body = f"""
  outs       <= not ins;
    """

    return generate_unary(
        name=name,
        modType="not",
        bitwidth=bitwidth,
        body=body,
        extra_signals=params.get("extra_signals", None),
    )
