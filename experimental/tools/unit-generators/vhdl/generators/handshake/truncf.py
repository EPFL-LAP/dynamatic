from generators.support.arith1 import generate_arith1


def generate_truncf(name, params):

    signals = f"""
  signal float_value : float64;
  signal float_truncated : float32;
    """

    body = f"""
  float_value <= to_float64(ins);
  float_truncated <= to_float32(float_value);
  outs <= to_std_logic_vector(float_truncated);
    """

    return generate_arith1(
        name=name,
        modType="truncf",
        input_bitwidth=32,
        output_bitwidth=64,
        signals=signals,
        body=body,
        extra_signals=params.get("extra_signals", None),
    )
