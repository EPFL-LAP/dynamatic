`timescale 1ns/1ps
// The original implementation has a parameter called INPUTS,
// used to indicate the number of input channels. I removed
// it. -- Jiantao, 21/07/2024
module br #(
	parameter DATA_TYPE = 32 // Default bit width set to 32
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

	br_dataless control (
		.clk        (clk       ),
		.rst        (rst       ),
		.ins_valid  (ins_valid ),
    .ins_ready  (ins_ready ),
		.outs_valid (outs_valid),
		.outs_ready (outs_ready)
	);
	
	assign outs = ins;

endmodule
