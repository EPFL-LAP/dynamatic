from generators.support.arith_binary import generate_arith_binary


def generate_minsi(name, params):
    bitwidth = params["bitwidth"]

    signals = f"""
        signal min_val_signed : signed(lhs'range);
    """

    body = f"""
        min_val_signed <= signed(lhs) when signed(lhs) < signed(rhs) else 
                          signed(rhs);
        result <= std_logic_vector(min_val_signed);
    """

    return generate_arith_binary(
        name=name,
        handshake_op="minsi",
        bitwidth=bitwidth,
        body=body,
        signals=signals,
        extra_signals=params.get("extra_signals", None)
    )
