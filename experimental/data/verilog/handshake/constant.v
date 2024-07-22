module ENTITY_NAME #(
	parameter BITWIDTH = 32 // Default set to 32 bits
)(
	input  clk,
	input  rst,
  // Input Channel
	input  ctrl_valid,
  output ctrl_ready,
  // Output Channel
  output [BITWIDTH - 1 : 0] outs,
  output outs_valid,
	input  outs_ready
);
	assign outs = "VALUE";	//! What is this? This is not a valid HDL file, Jiantao 21/07/2024 
	assign outs_valid = ctrl_valid;
	assign ctrl_ready = outs_ready;

endmodule
