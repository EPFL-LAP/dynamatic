from generators.handshake.dataless.dataless_br import generate_dataless_br
def generate_br(name, params):
    bitwidth = params["bitwidth"]

    dataless_br_name = name + "_dataless_br"
    verilog_header = "`timescale 1ns/1ps\n"
    verilog_dataless_br = generate_dataless_br(dataless_br_name, params)
    
    verilog_body_br = f"""
// Module of br
module {name} #(
	parameter DATA_TYPE = {bitwidth}
)(
	input clk,
	input rst,
  // Input Channel
	input [DATA_TYPE - 1 : 0] ins,   
	input ins_valid,
  output ins_ready,
  // Output Channel 	
  output [DATA_TYPE - 1 : 0] outs,
  output outs_valid	,
	input outs_ready			        
);

	{dataless_br_name} control (
		.clk        (clk       ),
		.rst        (rst       ),
		.ins_valid  (ins_valid ),
    .ins_ready  (ins_ready ),
		.outs_valid (outs_valid),
		.outs_ready (outs_ready)
	);
	
	assign outs = ins;

endmodule
"""

    return verilog_header + verilog_dataless_br + verilog_body_br