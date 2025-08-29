def generate_extui(name, params):
    output_type = params["output_type"]
    input_type = params["input_type"]

    verilog_extui = f"""
`timescale 1ns/1ps
// Module extui
module {name} #(
  parameter INPUT_TYPE = {input_type},
  parameter OUTPUT_TYPE = {output_type}
)(
  // inputs
  input  clk,
  input  rst,
  input  [INPUT_TYPE - 1 : 0] ins,
  input  ins_valid,
  output  ins_ready,
  // outputs
  output [OUTPUT_TYPE - 1 : 0] outs,
  output outs_valid,
  input outs_ready
);

  assign outs = {{{{(OUTPUT_TYPE - INPUT_TYPE){{1'b0}}}}, ins}};
  assign outs_valid = ins_valid;
  assign ins_ready = outs_ready;

endmodule
"""

    return verilog_extui
