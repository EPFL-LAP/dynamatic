`timescale 1ns/1ps
module tehb #(
	parameter DATA_TYPE = 32 // Default set to 32 bits
)(
	input  clk,
	input  rst,
  // Input Channel
	input  [DATA_TYPE - 1 : 0] ins,
	input  ins_valid,
  output ins_ready,
  // Output Channel
  output [DATA_TYPE - 1 : 0]	outs,
  output outs_valid,
	input  outs_ready
);
	// Signal Definition
	wire regEnable, regNotFull;
	reg [DATA_TYPE - 1 : 0] dataReg = 0;

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

	always @(posedge clk) begin
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
