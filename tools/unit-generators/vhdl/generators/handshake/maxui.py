from generators.support.arith_binary import generate_arith_binary


def generate_maxui(name, params):
    bitwidth = params["bitwidth"]

    signals = f"""
        signal max_val_unsigned : unsigned(lhs'range);
    """

    body = f"""
        max_val_unsigned <= unsigned(lhs) when unsigned(lhs) > unsigned(rhs) else unsigned(rhs);
        result <= std_logic_vector(max_val_unsigned);
    """

    return generate_arith_binary(
        name=name,
        handshake_op="maxui",
        bitwidth=bitwidth,
        signals=signals,
        body=body,
        extra_signals=params.get("extra_signals", None)
    )
