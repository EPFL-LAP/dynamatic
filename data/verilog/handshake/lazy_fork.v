`timescale 1ns/1ps
module lazy_fork #(
  parameter SIZE = 2,
  parameter DATA_TYPE = 32
)(
  input  clk,
	input  rst,
  // Input Channels
  input  [DATA_TYPE - 1 : 0] ins,
  input  ins_valid,
  output ins_ready,
  // Output Channel
  output [SIZE * (DATA_TYPE) - 1 : 0] outs,
	output [SIZE - 1 : 0] outs_valid,
	input  [SIZE - 1 : 0] outs_ready
);

  lazy_fork_dataless #(
    .SIZE(SIZE)
  ) control (
    .clk 			    (clk				        ),
    .rst 			    (rst				        ),
    .ins_valid 		(ins_valid			    ),
    .ins_ready    (ins_ready		      ),
    .outs_valid 	(outs_valid     	  ),
    .outs_ready 	(outs_ready         )
  );

  assign outs = {SIZE{ins}};

endmodule
