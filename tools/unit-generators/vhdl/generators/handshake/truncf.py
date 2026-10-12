from generators.support.unary import generate_unary


def generate_truncf(name, params):

    signals = f"""
  signal masked_ins : std_logic_vector(64 - 1 downto 0);
  signal float_value : float64;
  signal float_truncated : float32;
    """

    # 'to_float32' narrows a float64 and asserts fatally in simulation on an
    # out-of-range exponent. When 'ins_valid' is low the input is don't-care and
    # may carry a garbage double, so mask it to 0s before converting.
    body = f"""
  masked_ins <= ins when ins_valid = '1' else (others => '0');
  float_value <= to_float(masked_ins, 11, 52);
  float_truncated <= to_float32(float_value);
  outs <= to_std_logic_vector(float_truncated);
    """

    return generate_unary(
        name=name,
        handshake_op="truncf",
        input_bitwidth=64,
        output_bitwidth=32,
        signals=signals,
        body=body,
        extra_signals=params.get("extra_signals", None)
    )
