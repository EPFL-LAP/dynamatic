`timescale 1ns/1ps
module sink #(
  parameter DATA_WIDTH = 32
) (
  input  clk,      
  input  rst,       
  input  [DATA_WIDTH-1:0] ins, 
  input  ins_valid, 
  output ins_ready 
);
  assign ins_ready = 1'b1;

endmodule
