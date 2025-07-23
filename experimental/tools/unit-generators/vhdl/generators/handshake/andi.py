from generators.support.arith2 import generate_arith2

def generate_andi(name, params):
    bitwidth = params["bitwidth"]
    extra_signals = params.get("extra_signals", None)

    modType = "andi"

    signals = ""

    body = """
  result <= lhs and rhs;
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