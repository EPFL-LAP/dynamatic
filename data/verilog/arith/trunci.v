`timescale 1ns/1ps
module trunci #(
  parameter INPUT_TYPE = 32,
  parameter OUTPUT_TYPE = 32
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
  assign ins_ready = ~ins_valid | (ins_valid & outs_ready);

endmodule