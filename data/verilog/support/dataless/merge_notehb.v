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
	reg tmp_valid_out;
  reg [INPUTS - 1 : 0] tmp_ready_out;
	integer i;

	always @(*) begin
		tmp_valid_out = 0;
    tmp_ready_out = {INPUTS{1'b0}}; 
		for (i = 0; i < INPUTS; i = i + 1) begin
			if (ins_valid[i] && !tmp_valid_out) begin
				tmp_valid_out = 1;
        tmp_ready_out[i] = outs_ready;
			end
		end
	end
  
	assign	outs_valid  = tmp_valid_out;
	assign  ins_ready   = tmp_ready_out;

endmodule
