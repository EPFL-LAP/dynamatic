`timescale 1ns/1ps
module join_type #(
	parameter SIZE = 2 // Default Join input set to 2
)(
	input [SIZE - 1 : 0] ins_valid,
	input outs_ready,

	output reg  [SIZE - 1 : 0] ins_ready = 0,
	output outs_valid
);
	
	assign outs_valid = &ins_valid; // AND of all the bits in ins_valid vector
	
	reg [SIZE - 1 : 0] singleValid = 0;
	integer i, j;
	
	always @(*)begin
		for (i = 0; i < SIZE; i = i + 1) begin
			singleValid[i] = 1;
			for (j = 0; j < SIZE; j = j + 1)
				if (i != j)
					singleValid[i] = singleValid[i] & ins_valid[j];
		end
		
		for (i = 0; i < SIZE; i = i + 1) begin
			ins_ready[i] = singleValid[i] & outs_ready;
		end
	end
	
endmodule
