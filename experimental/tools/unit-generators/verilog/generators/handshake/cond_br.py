from generators.handshake.dataless.dataless_cond_br import generate_dataless_cond_br
def generate_cond_br(name, params):
    bitwidth = params["bitwidth"]

    if(bitwidth == 0):
        return generate_dataless_cond_br(name, {})

    dataless_cond_br_name = name + "_dataless_cond_br"
    header = "`timescale 1ns/1ps\n"
    dataless_cond_br = generate_dataless_cond_br(dataless_cond_br_name, {})

    body_cond_br = f"""
// Module of cond_br
module {name}(
	input clk,
	input rst,
  // Data Input Channel
	input [{bitwidth} - 1 : 0] data,				
	input data_valid,
  output data_ready,
  // Condition Input Channel
	input condition,		
	input condition_valid,
  output condition_ready,
  // True Output Channel
  output [{bitwidth} - 1 : 0] trueOut,
  output trueOut_valid,
	input trueOut_ready,
  // False Output Channel
  output [{bitwidth} - 1 : 0] falseOut,
  output falseOut_valid,
	input falseOut_ready
);
	{dataless_cond_br_name} control (
		.clk			       (clk			       ),
		.rst			       (rst	    	     ),
		.data_valid		   (data_valid	   ),
    .data_ready		   (data_ready	   ),
		.condition		   (condition		   ),
		.condition_valid (condition_valid),
    .condition_ready (condition_ready),
    .trueOut_valid	 (trueOut_valid	 ),
		.trueOut_ready	 (trueOut_ready	 ),
		.falseOut_valid	 (falseOut_valid ),
		.falseOut_ready	 (falseOut_ready )
	);

	assign trueOut = data;
	assign falseOut = data;

endmodule

"""

    return header + dataless_cond_br + body_cond_br