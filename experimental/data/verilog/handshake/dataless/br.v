`timescale 1ns/1ps
module br_dataless (
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
