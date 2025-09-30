from generators.support.arith_binary import generate_arith_binary

def generate_shli(name, params):

  bitwidth = params["bitwidth"]

  body = "assign result = lhs << rhs;"

  return generate_arith_binary(
    name=name,
    op_body=body,
    handshake_op="shli",
    bitwidth=bitwidth,
    extra_signals=params.get("extra_signals", None))
