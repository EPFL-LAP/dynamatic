from generators.support.arith1 import generate_arith1


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

    return generate_arith1(
        name=name,
        modType="extf",
        input_bitwidth=32,
        output_bitwidth=64,
        signals=signals,
        body=body,
        extra_signals=params.get("extra_signals", None),
    )
