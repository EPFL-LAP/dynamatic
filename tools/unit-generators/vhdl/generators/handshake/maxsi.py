from generators.support.arith_binary import generate_arith_binary


def generate_maxsi(name, params):
    bitwidth = params["bitwidth"]

    signals = f"""
        signal max_val_signed : signed(lhs'range);
    """

    body = f"""
        max_val_signed <= signed(lhs) when signed(lhs) > signed(rhs) else 
                          signed(rhs);
        result <= std_logic_vector(max_val_signed);
    """

    return generate_arith_binary(
        name=name,
        handshake_op="maxsi",
        bitwidth=bitwidth,
        body=body,
        signals=signals,
        extra_signals=params.get("extra_signals", None)
    )
