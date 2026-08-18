from generators.support.unary import generate_unary


def generate_absi(name, params):
    bitwidth = params["bitwidth"]

    body = f"""
  outs({bitwidth} - 1 downto 0) <= std_logic_vector(abs(signed(ins)));
    """

    return generate_unary(
        name=name,
        handshake_op="absi",
        bitwidth=bitwidth,
        body=body,
        extra_signals=params.get("extra_signals", None),
    )
