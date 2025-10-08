from generators.support.signal_manager import generate_default_signal_manager
from generators.handshake.join import generate_join

def generate_cond_br(name, params):
    bitwidth = params["bitwidth"]
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_cond_br_signal_manager(name, bitwidth, extra_signals)
    elif bitwidth == 0:
        return _generate_cond_br_dataless(name)
    else:
        return _generate_cond_br(name, bitwidth)

def _generate_cond_br(name, bitwidth):

    cond_br_dataless_name = name + "_cond_br_dataless"
    cond_br_dataless = _generate_cond_br_dataless(cond_br_dataless_name)

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
	{cond_br_dataless_name} control (
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

    return cond_br_dataless + body_cond_br

def _generate_cond_br_dataless(name):

    join_name = name + "_join"

    join_instance = generate_join(join_name, {"size":2})

    body_cond_br_dataless = f"""
// Module of cond_br_dataless

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

    return join_instance + body_cond_br_dataless

def _generate_cond_br_signal_manager(name, bitwidth, extra_signals):
    return generate_default_signal_manager(
        name,
        [{
            "name": "data",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }, {
            "name": "condition",
            "bitwidth": 1,
            "extra_signals": extra_signals
        }],
        [{
            "name": "trueOut",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }, {
            "name": "falseOut",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }],
        extra_signals,
        lambda name:
            (_generate_cond_br_dataless(name) if bitwidth == 0
             else _generate_cond_br(name, bitwidth)))
