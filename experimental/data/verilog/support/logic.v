`timescale 1ns/1ps
//-----------------------------------------------------------------------
//-- logical and, version 0.1, 21/07/2024, Jiantao
//-----------------------------------------------------------------------
module and_n #(
	parameter SIZE = 2	// Default set to 2 inputs
)(
	input [SIZE - 1 : 0] ins,
	
	output outs
);
	// Generate a constant vector of ones
	wire [SIZE - 1 : 0] all_ones = {SIZE{1'b1}};

	assign outs = (ins == all_ones) ? 1'b1 : 1'b0;

endmodule


//-----------------------------------------------------------------------
//-- logical nand, version 0.1, 21/07/2024, Jiantao
//-----------------------------------------------------------------------
module nand_n #(
	parameter SIZE = 2
)(
	input [SIZE - 1 : 0] ins,

	output outs
);
	wire [SIZE - 1 : 0] all_ones = {SIZE{1'b1}};

	assign outs = (ins == all_ones) ? 1'b0 : 1'b1;

endmodule

//-----------------------------------------------------------------------
//-- logical or, version 0.1, 21/07/2024, Jiantao
//-----------------------------------------------------------------------
module or_n #(
	parameter SIZE = 2
)(
	input [SIZE - 1 : 0] ins,

	output outs
);
	wire [SIZE - 1 : 0] all_zeros = {SIZE{1'b0}};

	assign outs = (ins == all_zeros) ? 1'b0 : 1'b1;

endmodule

//-----------------------------------------------------------------------
//-- logical nor, version 0.1, 21/07/2024, Jiantao
//-----------------------------------------------------------------------
module nor_n #(
	parameter SIZE = 2
)(
	input [SIZE - 1 : 0] ins,

	output outs
);
	wire [SIZE - 1 : 0] all_zeros = {SIZE{1'b0}};

	assign outs = (ins == all_zeros) ? 1'b1 : 1'b0;

endmodule
