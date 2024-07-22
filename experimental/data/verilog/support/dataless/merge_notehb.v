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
		outs_valid = tmp_valid_out;
	end

	// Distribute the ready signal to all input channels
	// TODO: Simplify the following always block, not needed I think
	always @(*) begin
		for (i = 0; i < INPUTS; i = i + 1) begin
			ins_ready[i] = outs_ready;
		end
	end

endmodule
