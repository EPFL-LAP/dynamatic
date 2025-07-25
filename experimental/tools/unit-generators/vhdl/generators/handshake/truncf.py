from generators.support.arith1 import generate_arith1


def generate_truncf(name, params):

    signals = f"""
  signal float_value : float64;
  signal float_truncated : float32;
    """

    body = f"""
  float_value <= to_float(ins, 11, 52);
  float_truncated <= to_float32(float_value);
  outs <= to_std_logic_vector(float_truncated);
    """

    return generate_arith1(
        name=name,
        modType="truncf",
        input_bitwidth=64,
        output_bitwidth=32,
        signals=signals,
        body=body,
        extra_signals=params.get("extra_signals", None),
    )
