from generators.support.arith1 import generate_arith1


def generate_absf(name, params):
    bitwidth = params["bitwidth"]

    body = f"""
  outs({bitwidth} - 1)          <= '0';
  outs({bitwidth} - 2 downto 0) <= ins({bitwidth} - 2 downto 0);
    """

    return generate_arith1(
        name=name,
        modType="absf",
        bitwidth=bitwidth,
        body=body,
        extra_signals=params.get("extra_signals", None),
    )
