`timescale 1ns/1ps
// In the original implementation
// Data Organization: out2-:32, out1+:32
// condition is implemeted as [0:0], I removed the redudancy
module cond_br_dataless (
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
	assign ins_valid_vec = {condition_valid, data_valid};

	join_type #(
		.SIZE(2)
	) join_branch (
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
