from generators.support.arith1 import generate_arith1


def generate_trunci(name, params):
    input_bitwidth = params["input_bitwidth"]
    output_bitwidth = params["output_bitwidth"]
    extra_signals = params.get("extra_signals", None)

    modType = "trunci"

    body = f"""
  outs       <= ins({output_bitwidth} - 1 downto 0);
  outs_valid <= ins_valid;
  ins_ready  <= not ins_valid or (ins_valid and outs_ready);
    """

    return generate_arith1(
        name=name,
        modType=modType,
        input_bitwidth=input_bitwidth,
        output_bitwidth=output_bitwidth,
        body=body,
        extra_signals=extra_signals,
    )
