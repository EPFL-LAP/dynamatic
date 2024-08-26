`timescale 1ns/1ps
module negf #(
  parameter DATA_TYPE = 32
)(
  // inputs
  input  clk,
  input  rst,
  input  [DATA_TYPE - 1 : 0] ins,
  input  ins_valid,
  input  ins_ready,
  // outputs
  output [DATA_TYPE - 1 : 0] outs,
  output outs_valid,
  output outs_ready
);

  assign outs[DATA_TYPE-1] = ins[DATA_TYPE-1] ^ 1'b1;
  assign outs[DATA_TYPE-2:0] = ins[DATA_TYPE-2:0];
  assign outs_valid = ins_valid;
  assign ins_ready = outs_ready;

endmodule