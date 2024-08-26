`timescale 1ns/1ps
module extsi #(
  parameter INPUT_TYPE = 32,
  parameter OUTPUT_TYPE = 64
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

  assign outs = {{(OUTPUT_TYPE - INPUT_TYPE){ins[INPUT_TYPE - 1]}}, ins};
  assign outs_valid = ins_valid;
  assign ins_ready = outs_ready;

endmodule