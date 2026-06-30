from generators.support.unary import generate_unary


def generate_extf(name, params):

    signals = f"""
  signal float_value : float32;
  signal float_extended : float64;
    """

    body = f"""
  float_value     <= to_float(ins);
  float_extended  <= to_float64(float_value);
  outs            <= to_std_logic_vector(float_extended);
    """

    return generate_unary(
        name=name,
        handshake_op="extf",
        input_bitwidth=32,
        output_bitwidth=64,
        signals=signals,
        body=body,
        extra_signals=params.get("extra_signals", None),
    )
