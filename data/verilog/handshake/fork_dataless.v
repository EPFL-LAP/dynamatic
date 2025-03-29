`timescale 1ns/1ps
module fork_dataless #(
	parameter SIZE = 2
)(
	input  clk,
	input  rst,
  // Input Channel
	input  ins_valid,
  output ins_ready,
  // Output Channel
  output [SIZE - 1 : 0] outs_valid,
	input  [SIZE - 1: 0] outs_ready
);
	// Internal Signal Definition
	wire [SIZE - 1 : 0] blockStopArray;
	wire anyBlockStop;
	wire backpressure;

	or_n #(
		.SIZE(SIZE)
	) anyBlockFull (
		.ins  (blockStopArray),
		.outs (anyBlockStop  )
	);

	assign ins_ready = ~anyBlockStop;
	assign backpressure = ins_valid & anyBlockStop;

	// Define generate variable
	genvar gen;

	generate
		for (gen = SIZE - 1; gen >= 0; gen = gen - 1) begin: regBlock
			eager_fork_register_block regblock (
				.clk 			    (clk				        ),
				.rst 			    (rst				        ),
				.ins_valid 		(ins_valid			    ),
				.outs_ready 	(outs_ready[gen]	  ),
				.backpressure (backpressure		    ),
				.outs_valid 	(outs_valid[gen]	  ),
				.blockStop 		(blockStopArray[gen])
			);
		end
	endgenerate


endmodule
