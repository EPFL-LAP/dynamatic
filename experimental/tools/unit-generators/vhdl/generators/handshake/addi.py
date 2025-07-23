from generators.support.signal_manager import generate_arith2_signal_manager
from generators.support.arith2 import generate_arith2

def generate_addi(name, params):
    bitwidth = params["bitwidth"]
    extra_signals = params.get("extra_signals", None)

    modType = "addi"

    signals = ""

    body = """
  result <= std_logic_vector(unsigned(lhs) + unsigned(rhs));
    """

    dependencies = ""
    latency = 0

    return generate_arith2(
          name,
          modType,
          bitwidth,
          signals,
          body,
          latency,
          dependencies,
          extra_signals,
      )