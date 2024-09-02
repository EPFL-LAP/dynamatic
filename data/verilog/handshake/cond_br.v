`timescale 1ns/1ps
module cond_br #(
	parameter DATA_TYPE = 32 // Default set to 32 bits
)(
	input clk,
	input rst,
  // Data Input Channel
	input [DATA_TYPE - 1 : 0] data,				
	input data_valid,
  output data_ready,
  // Condition Input Channel
	input condition,		
	input condition_valid,
  output condition_ready,
  // True Output Channel
  output [DATA_TYPE - 1 : 0] trueOut,
  output trueOut_valid,
	input trueOut_ready,
  // False Output Channel
  output [DATA_TYPE - 1 : 0] falseOut,
  output falseOut_valid,
	input falseOut_ready
);
	cond_br_dataless control (
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
