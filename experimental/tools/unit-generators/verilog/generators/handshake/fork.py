def generate_fork(name, params):
    size = params["size"]
    bitwidth = params["bitwidth"]

    if(bitwidth == 0):
        return generate_datalessFork(name, {"size": size})
    
    datalessFork_name = name + "_datalessFork"



    datalessFork = generate_datalessFork(datalessFork_name, {"size": size})
    Fork = f"""
// Module of Fork
module {name}(
	input  clk,
	input  rst,
  // Input Channel
	input  [{bitwidth} - 1 : 0] ins,
	input  ins_valid,
  output ins_ready,
  // Output Channel
  output [{size} * ({bitwidth}) - 1 : 0] outs,
  output [{size} - 1 : 0] outs_valid,
	input  [{size} - 1 : 0] outs_ready
);

  {datalessFork_name} control (
    .clk        (clk        ),
    .rst        (rst        ),
    .ins_valid  (ins_valid  ),
    .ins_ready  (ins_ready  ),
    .outs_valid (outs_valid ),
    .outs_ready (outs_ready )
  );

  // Broadcast the input data to all output channels
  assign outs = {{{size}{{ins}}}};

endmodule
"""


    return datalessFork + Fork

def generate_datalessFork(name, params):
    size = params["size"]

    eager_fork_register_block_name = name + "_eager_fork_register_block"
    eager_fork_register_block = f"""
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

    datalessFork = f"""
// Module of datalessFork

module {name}(
	input  clk,
	input  rst,
  // Input Channel
	input  ins_valid,
  output ins_ready,
  // Output Channel
  output [{size} - 1 : 0] outs_valid,
	input  [{size} - 1: 0] outs_ready
);
	// Internal Signal Definition
	wire [{size} - 1 : 0] blockStopArray;
	wire anyBlockStop;
	wire backpressure;

	assign anyBlockStop = |blockStopArray;

	assign ins_ready = ~anyBlockStop;
	assign backpressure = ins_valid & anyBlockStop;

	// Define generate variable
	genvar gen;

	generate
		for (gen = {size} - 1; gen >= 0; gen = gen - 1) begin: regBlock
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


    return eager_fork_register_block + datalessFork
