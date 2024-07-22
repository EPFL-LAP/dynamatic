module sink #(
  parameter BITWIDTH = 32
) (
  input  clk,      
  input  rst,       
  input  [BITWIDTH-1:0] ins, 
  input  ins_valid, 
  output ins_ready 
);
  assign ins_ready = 1'b1;

endmodule
