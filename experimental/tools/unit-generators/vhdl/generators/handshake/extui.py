from generators.support.arith1 import generate_arith1


def generate_extui(name, params):
    input_bitwidth = params["input_bitwidth"]
    output_bitwidth = params["output_bitwidth"]

    body = f"""
  outs({output_bitwidth} - 1 downto {input_bitwidth}) <= ({output_bitwidth} - {input_bitwidth} - 1 downto 0 => '0');
  outs({input_bitwidth} - 1 downto 0)            <= ins;
    """

    return generate_arith1(
        name=name,
        modType="extui",
        input_bitwidth=input_bitwidth,
        output_bitwidth=output_bitwidth,
        body=body,
        extra_signals=params.get("extra_signals", None),
    )
