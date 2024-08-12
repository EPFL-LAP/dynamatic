`timescale 1ns/1ps
module merge_notehb_dataless #(
	parameter INPUTS = 2 // Default set to 2 inputs
)(
	input  clk,
	input  rst,
  // Input Channels
	input  [INPUTS - 1 : 0] ins_valid,
  output [INPUTS - 1 : 0] ins_ready,
  // Output Channels
  output outs_valid,
	input  outs_ready
	
);
	reg tmp_valid_out = 0;

	// Define iteration variable
	integer i;

	always @(*) begin
		tmp_valid_out = 0;
		for (i = INPUTS - 1; i >= 0; i = i - 1) begin
			if (ins_valid[i]) begin
				tmp_valid_out = 1;
			end
		end
	end
	assign	outs_valid = tmp_valid_out;

	// Distribute the ready signal to all input channels
	assign ins_ready = {INPUTS{outs_ready}};

endmodule
