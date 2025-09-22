from generators.handshake.dataless.dataless_br import generate_dataless_br
def generate_br(name, params):
    bitwidth = params["bitwidth"]

    dataless_br_name = name + "_dataless_br"
    header = "`timescale 1ns/1ps\n"
    dataless_br = generate_dataless_br(dataless_br_name, {})
    
    body_br = f"""
// Module of br
module {name}(
	input clk,
	input rst,
  // Input Channel
	input [{bitwidth} - 1 : 0] ins,   
	input ins_valid,
  output ins_ready,
  // Output Channel 	
  output [{bitwidth} - 1 : 0] outs,
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

    return header + dataless_br + body_br