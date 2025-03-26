`timescale 1ns/1ps
module sink_dataless (
    input  clk,    
    input  rst,     
    input  ins_valid,  
    output ins_ready 
);
  assign ins_ready = 1'b1;

endmodule