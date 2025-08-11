from generators.support.unary import generate_unary


def generate_extsi(name, params):
    input_bitwidth = params["input_bitwidth"]
    output_bitwidth = params["output_bitwidth"]

    body = f"""
  outs({output_bitwidth} - 1 downto {input_bitwidth}) <= ({output_bitwidth} - {input_bitwidth} - 1 downto 0 => ins({input_bitwidth} - 1));
  outs({input_bitwidth} - 1 downto 0)            <= ins;
    """

    return generate_unary(
        name=name,
        modType="extsi",
        input_bitwidth=input_bitwidth,
        output_bitwidth=output_bitwidth,
        body=body,
        extra_signals=params.get("extra_signals", None),
    )
