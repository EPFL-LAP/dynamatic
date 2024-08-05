`timescale 1ns/1ps
module negf #(
  parameter DATA_WIDTH = 32
)(
  // inputs
  input  clk,
  input  rst,
  input  [DATA_WIDTH - 1 : 0] ins,
  input  ins_valid,
  input  ins_ready,
  // outputs
  output [DATA_WIDTH - 1 : 0] outs,
  output outs_valid,
  output outs_ready
);

  assign outs[DATA_WIDTH-1] = ins[DATA_WIDTH-1] ^ 1'b1;
  assign outs[DATA_WIDTH-2:0] = ins[DATA_WIDTH-2:0];
  assign outs_valid = ins_valid;
  assign ins_ready = outs_ready;

endmodule