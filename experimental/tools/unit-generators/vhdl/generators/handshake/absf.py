from generators.support.unary import generate_unary


def generate_absf(name, params):
    bitwidth = params["bitwidth"]

    body = f"""
  outs({bitwidth} - 1)          <= '0';
  outs({bitwidth} - 2 downto 0) <= ins({bitwidth} - 2 downto 0);
    """

    return generate_unary(
        name=name,
        op_type="absf",
        bitwidth=bitwidth,
        body=body,
        extra_signals=params.get("extra_signals", None),
    )
