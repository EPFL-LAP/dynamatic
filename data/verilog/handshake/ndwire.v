`timescale 1ns/1ps
module ndwire #(
  parameter DATA_TYPE = 32
) (
  input  clk,
  input  rst,
  // Random stall
  input stall,
  // Input channel
  input  [DATA_TYPE - 1 : 0] ins,
  input  ins_valid,
  output ins_ready,
  // Output channel
  output [DATA_TYPE - 1 : 0] outs,
  output outs_valid,
  input  outs_ready
);

  assign ins_ready = outs_ready && !stall;
  assign outs_valid = ins_valid && !stall;
  assign outs = ins;

endmodule
