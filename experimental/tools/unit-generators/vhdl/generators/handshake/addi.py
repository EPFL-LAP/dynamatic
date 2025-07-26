from generators.support.arith2 import generate_arith2


def generate_addi(name, params):
    bitwidth = params["bitwidth"]

    body = """
  result <= std_logic_vector(unsigned(lhs) + unsigned(rhs));
    """

    return generate_arith2(
        name=name,
        modType="addi",
        bitwidth=bitwidth,
        body=body,
        extra_signals=params.get("extra_signals", None),
    )
