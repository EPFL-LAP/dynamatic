from generators.handshake.binop import generate_binop

def generate_andi(name, params):

  bitwidth = params["bitwidth"]

  body = "assign result = lhs & rhs;"

  return generate_binop(
    name=name,
    op_body=body,
    handshake_op="andi",
    bitwidth=bitwidth,
    extra_signals=params.get("extra_signals", None))
