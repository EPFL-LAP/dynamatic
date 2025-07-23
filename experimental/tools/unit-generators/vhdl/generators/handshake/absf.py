from generators.support.arith1 import generate_arith1

def generate_absf(name, params):
    bitwidth = params["bitwidth"]
    extra_signals = params.get("extra_signals", None)


    modType = "absf"
    signals = ""

    body = f"""
  outs({bitwidth} - 1)          <= '0';
  outs({bitwidth} - 2 downto 0) <= ins({bitwidth} - 2 downto 0);
  outs_valid                  <= ins_valid;
  ins_ready                   <= outs_ready;
    """

    dependencies = ""
    latency = 0

    return generate_arith1(
          name,
          modType,
          bitwidth,
          signals,
          body,
          dependencies,
          latency,
          extra_signals,
      )