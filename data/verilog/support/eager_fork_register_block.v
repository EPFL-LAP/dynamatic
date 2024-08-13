`timescale 1ns/1ps
module eager_fork_register_block (
	input clk,
	input rst,

	input ins_valid,	// Input Channel
	input outs_ready,	// Output Channel
	input backpressure,

	output outs_valid,
	output blockStop
);
	reg transmitValue = 0;
	wire keepValue;

	assign keepValue = ~outs_ready & transmitValue;

	always @(posedge clk, posedge rst) begin
		if (rst) begin
			transmitValue <= 1;
		end else begin
			transmitValue <= keepValue | ~backpressure;
		end
	end

	assign outs_valid = transmitValue & ins_valid;
	assign blockStop = keepValue;

endmodule
