from generators.handshake.binop import generate_binop

def generate_shrsi(name, params):
  bitwidth = params["bitwidth"]
  body = f"""
    wire signed [{bitwidth} - 1 : 0] signed_lhs;
    wire signed [{bitwidth} - 1 : 0] temp_result;
    assign signed_lhs = lhs;
    assign temp_result = signed_lhs >>> rhs;
  
    assign result = temp_result;
  """

  return generate_binop(
    name=name,
    op_body=body,
    handshake_op="shrsi",
    bitwidth=bitwidth,
    extra_signals=params.get("extra_signals", None))
