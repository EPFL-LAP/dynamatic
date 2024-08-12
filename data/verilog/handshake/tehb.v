`timescale 1ns/1ps
// Jiantao, 21/07/2024
// The original implementation has two more parameters INPUTS and OUTPUTs, removed
//! Verilog implementation is wrong for TEHB, the logic in line 84 is wrong
module tehb #(
	parameter DATA_WIDTH = 32 // Default set to 32 bits
)(
	input  clk,
	input  rst,
  // Input Channel
	input  [DATA_WIDTH - 1 : 0] ins,
	input  ins_valid,
  output ins_ready,
  // Output Channel
  output [DATA_WIDTH - 1 : 0]	outs,
  output outs_valid,
	input  outs_ready
);
	// Signal Definition
	wire regEnable, regNotFull;
	reg [DATA_WIDTH - 1 : 0] dataReg = 0;

	// Instantiate control logic part
	tehb_dataless control (
		.clk		    (clk	     ),
		.rst		    (rst	     ),
		.ins_valid	(ins_valid ),
    .ins_ready	(regNotFull),
		.outs_valid	(outs_valid),
		.outs_ready	(outs_ready)
	);

	assign regEnable = regNotFull & ins_valid & ~outs_ready;

	always @(posedge clk, posedge rst) begin
		if (rst) begin
			dataReg <= 0;
		end else if (regEnable) begin
			dataReg <= ins;
		end
	end

	// Output Assignment
	assign outs = regNotFull ? ins : dataReg;

	assign ins_ready = regNotFull;

endmodule
