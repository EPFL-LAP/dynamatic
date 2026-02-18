from generators.support.unary import generate_unary


def generate_noti(name, params):
    bitwidth = params["bitwidth"]

    body = f"""
  outs       <= not ins;
    """

    return generate_unary(
        name=name,
        handshake_op="noti",
        bitwidth=bitwidth,
        body=body,
        extra_signals=params.get("extra_signals", None),
    )
