def generate_trunci(name, params):
    output_type = params["output_type"]
    input_type = params["input_type"]

    extui = f"""
`timescale 1ns/1ps
// Module of trunci
module {name} #(
  parameter INPUT_TYPE = {input_type},
  parameter OUTPUT_TYPE = {output_type}
)(
  // inputs
  input  clk,
  input  rst,
  input  [INPUT_TYPE - 1 : 0] ins,
  input  ins_valid,
  input  outs_ready,
  // outputs
  output [OUTPUT_TYPE - 1 : 0] outs,
  output outs_valid,
  output ins_ready
);

  assign outs = ins[OUTPUT_TYPE - 1 : 0];
  assign outs_valid = ins_valid;
  assign ins_ready = ins_valid & outs_ready;

endmodule
"""

    return extui
