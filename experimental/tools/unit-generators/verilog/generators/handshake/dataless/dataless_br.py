def generate_dataless_br(name, params):

    body_dataless_br = f"""
`timescale 1ns/1ps
//dataless_br Module
module {name} (
	input  clk,
	input  rst,
  // Input Channel
	input  ins_valid,
  output ins_ready,
  // Output Channel
  output outs_valid,
	input  outs_ready
);

	assign outs_valid = ins_valid;
	assign ins_ready = outs_ready;

endmodule
"""

    return body_dataless_br