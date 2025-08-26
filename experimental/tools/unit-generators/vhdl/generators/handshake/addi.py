from generators.support.arith2 import generate_arith_binary


def generate_addi(name, params):
    bitwidth = params["bitwidth"]

    body = """
  result <= std_logic_vector(unsigned(lhs) + unsigned(rhs));
    """

    return generate_arith_binary(
        name=name,
        op_type="addi",
        bitwidth=bitwidth,
        body=body,
        extra_signals=params.get("extra_signals", None),
    )
