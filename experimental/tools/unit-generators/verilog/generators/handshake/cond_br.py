from generators.handshake.join import generate_join
def generate_cond_br(name, params):
    bitwidth = params["bitwidth"]

    if(bitwidth == 0):
        return generate_dataless_cond_br(name, {})

    dataless_cond_br_name = name + "_dataless_cond_br"
    dataless_cond_br = generate_dataless_cond_br(dataless_cond_br_name, {})

    body_cond_br = f"""
// Module of cond_br
module {name}(
	input clk,
	input rst,
  // Data Input Channel
	input [{bitwidth} - 1 : 0] data,				
	input data_valid,
  output data_ready,
  // Condition Input Channel
	input condition,		
	input condition_valid,
  output condition_ready,
  // True Output Channel
  output [{bitwidth} - 1 : 0] trueOut,
  output trueOut_valid,
	input trueOut_ready,
  // False Output Channel
  output [{bitwidth} - 1 : 0] falseOut,
  output falseOut_valid,
	input falseOut_ready
);
	{dataless_cond_br_name} control (
		.clk			       (clk			       ),
		.rst			       (rst	    	     ),
		.data_valid		   (data_valid	   ),
    .data_ready		   (data_ready	   ),
		.condition		   (condition		   ),
		.condition_valid (condition_valid),
    .condition_ready (condition_ready),
    .trueOut_valid	 (trueOut_valid	 ),
		.trueOut_ready	 (trueOut_ready	 ),
		.falseOut_valid	 (falseOut_valid ),
		.falseOut_ready	 (falseOut_ready )
	);

	assign trueOut = data;
	assign falseOut = data;

endmodule

"""

    return dataless_cond_br + body_cond_br

def generate_dataless_cond_br(name, params):

    join_name = name + "_join"

    join_instance = generate_join(join_name, {"size":2})

    body_dataless_cond_br = f"""
// Module of dataless_cond_br

// In the original implementation
// Data Organization: out2-:32, out1+:32
// condition is implemeted as [0:0], I removed the redudancy
module {name} (
	input  clk,
	input  rst,
  // Data Input Channel
	input  data_valid,
  output data_ready,
  // Condition Input Channel
	input  condition,		   
	input  condition_valid,
  output condition_ready,
  // True Output Channel, Channel 1
  output trueOut_valid,
	input  trueOut_ready,
  // False Output Channel, Channel 2
  output falseOut_valid,
	input  falseOut_ready
);	
	wire [1 : 0] ins_valid_vec; // Vector for the join's ins_valid
	wire [1 : 0] ins_ready_vec; // Vector for the join's ins_ready
	
	wire branchInputs_valid;
	wire branch_ready;

	// Assign individual signals to the vector
	assign ins_valid_vec = {{condition_valid, data_valid}};

	{join_name} join_branch (
		.ins_valid  (ins_valid_vec     ),
		.outs_ready (branch_ready      ),
		.ins_ready  (ins_ready_vec     ),
		.outs_valid (branchInputs_valid)
	);

	// Connect the outputs of the join module to appropriate external signals
	assign data_ready = ins_ready_vec[0];
	assign condition_ready = ins_ready_vec[1];

	assign trueOut_valid  = condition & branchInputs_valid;
	assign falseOut_valid = ~condition & branchInputs_valid;
	assign branch_ready = (falseOut_ready & ~condition) | (trueOut_ready & condition);

endmodule
"""

    return join_instance + body_dataless_cond_br
