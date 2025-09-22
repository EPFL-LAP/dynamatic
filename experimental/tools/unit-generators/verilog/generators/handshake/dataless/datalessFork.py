def generate_datalessFork(name, params):
    size = params["size"]

    verilog_header = "`timescale 1ns/1ps\n"
    eager_fork_register_block_name = name + "_eager_fork_register_block"
    verilog_eager_fork_register_block = f"""
// Module of eager_fork_register_block
module {eager_fork_register_block_name} (
	input clk,
	input rst,

	input ins_valid,	// Input Channel
	input outs_ready,	// Output Channel
	input backpressure,

	output outs_valid,
	output blockStop
);
	reg transmitValue = 1;
	wire keepValue;

	assign keepValue = ~outs_ready & transmitValue;

	always @(posedge clk) begin
		if (rst) begin
			transmitValue <= 1;
		end else begin
			transmitValue <= keepValue | ~backpressure;
		end
	end

	assign outs_valid = transmitValue & ins_valid;
	assign blockStop = keepValue;

endmodule
"""

    verilog_datalessFork = f"""
// Module of datalessFork

module {name} #(
	parameter SIZE = {size}
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

	assign anyBlockStop = |blockStopArray;

	assign ins_ready = ~anyBlockStop;
	assign backpressure = ins_valid & anyBlockStop;

	// Define generate variable
	genvar gen;

	generate
		for (gen = SIZE - 1; gen >= 0; gen = gen - 1) begin: regBlock
			{eager_fork_register_block_name} regblock (
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

"""


    return verilog_header + verilog_eager_fork_register_block + verilog_datalessFork
