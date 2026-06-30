from generators.support.unary import generate_unary


def generate_negf(name, params):
    bitwidth = params.get("bitwidth", None)

    body = f"""
  outs({bitwidth} - 1)          <= ins({bitwidth} - 1) xor '1';
  outs({bitwidth} - 2 downto 0) <= ins({bitwidth} - 2 downto 0);
    """

    return generate_unary(
        name=name,
        handshake_op="negf",
        bitwidth=bitwidth,
        body=body,
        extra_signals=params.get("extra_signals", None)
    )
