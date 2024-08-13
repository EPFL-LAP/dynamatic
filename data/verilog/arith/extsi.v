`timescale 1ns/1ps
module extsi #(
  parameter INPUT_WIDTH = 32,
  parameter OUTPUT_WIDTH = 64
)(
  // inputs
  input  clk,
  input  rst,
  input  [INPUT_WIDTH - 1 : 0] ins,
  input  ins_valid,
  output  ins_ready,
  // outputs
  output [OUTPUT_WIDTH - 1 : 0] outs,
  output outs_valid,
  input outs_ready
);

  assign outs = {{(OUTPUT_WIDTH - INPUT_WIDTH){ins[INPUT_WIDTH - 1]}}, ins};
  assign outs_valid = ins_valid;
  assign ins_ready = outs_ready;

endmodule