def generate_source(name, params):
    return f"""
`timescale 1ns/1ps
// Module of source
module {name} (
  input  clk,        
  input  rst,        
  input  outs_ready, 
  output outs_valid 
);
  assign outs_valid = 1;

endmodule

"""