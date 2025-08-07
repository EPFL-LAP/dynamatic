from generators.support.unary import generate_unary


def generate_trunci(name, params):
    input_bitwidth = params["input_bitwidth"]
    output_bitwidth = params["output_bitwidth"]
    extra_signals = params.get("extra_signals", None)

    modType = "trunci"

    body = f"""
  outs       <= ins({output_bitwidth} - 1 downto 0);
    """

    return generate_unary(
        name=name,
        modType=modType,
        input_bitwidth=input_bitwidth,
        output_bitwidth=output_bitwidth,
        body=body,
        extra_signals=extra_signals
    )
