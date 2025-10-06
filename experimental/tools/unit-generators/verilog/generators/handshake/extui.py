def generate_extui(name, params):
    output_type = params["output_type"]
    input_type = params["input_type"]

    extui = f"""
// Module of extui
module {name}(
  // inputs
  input  clk,
  input  rst,
  input  [{input_type} - 1 : 0] ins,
  input  ins_valid,
  output  ins_ready,
  // outputs
  output [{output_type} - 1 : 0] outs,
  output outs_valid,
  input outs_ready
);

  assign outs = {{{{({output_type} - {input_type}){{1'b0}}}}, ins}};
  assign outs_valid = ins_valid;
  assign ins_ready = outs_ready;

endmodule
"""

    return extui
