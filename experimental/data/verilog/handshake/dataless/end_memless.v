`timescale 1ns/1ps
module end_sync_memless_dataless (
  input  clk,
  input  rst,
  // Input Channel
  input  [0 : 0] ins_valid,
  output [0 : 0] ins_ready,
  // Output Channel
  input  [0 : 0] outs_ready,
  output [0 : 0] outs_valid
);
  assign outs_valid[0] = ins_valid[0];
  assign ins_ready[0] = ins_valid[0] & outs_ready[0];

endmodule
