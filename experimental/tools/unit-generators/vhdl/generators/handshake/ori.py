from generators.support.arith2 import generate_arith_binary


def generate_ori(name, params):
    bitwidth = params["bitwidth"]

    body = """
  result <= lhs or rhs;
    """

    return generate_arith_binary(
        name=name,
        op_type="ori",
        bitwidth=bitwidth,
        body=body,
        extra_signals=params.get("extra_signals", None)
    )
