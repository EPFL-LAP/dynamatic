`timescale 1ns/1ps
module source (
  input  clk,        
  input  rst,        
  input  outs_ready, 
  output outs_valid 
);
  assign outs_valid = 1;

endmodule
