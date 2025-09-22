def generate_sink(name, params):
    bitwidth = params["bitwidth"]

    empty_sink = f"""
`timescale 1ns/1ps
// Module of sink
module {name} (
    input  clk,    
    input  rst,     
    input  ins_valid,  
    output ins_ready 
);
  assign ins_ready = 1'b1;

endmodule
"""
    non_empty_sink = f"""
`timescale 1ns/1ps
// Module of sink
module {name}(
  input  clk,      
  input  rst,       
  input  [{bitwidth}-1:0] ins, 
  input  ins_valid, 
  output ins_ready 
);
  assign ins_ready = 1'b1;

endmodule

"""

    if bitwidth:
        return non_empty_sink
    else:
        return empty_sink