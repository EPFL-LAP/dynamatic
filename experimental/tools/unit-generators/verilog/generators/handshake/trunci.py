def generate_trunci(name, params):
    output_type = params["output_type"]
    input_type = params["input_type"]

    extui = f"""
// Module of trunci
module {name}(
  // inputs
  input  clk,
  input  rst,
  input  [{input_type} - 1 : 0] ins,
  input  ins_valid,
  input  outs_ready,
  // outputs
  output [{output_type} - 1 : 0] outs,
  output outs_valid,
  output ins_ready
);

  assign outs = ins[{output_type} - 1 : 0];
  assign outs_valid = ins_valid;
  assign ins_ready = ins_valid & outs_ready;

endmodule
"""

    return extui
