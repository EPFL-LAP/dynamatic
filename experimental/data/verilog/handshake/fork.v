module fork #(
	parameter SIZE = 2,
	parameter BITWIDTH = 32
)(
	input  clk,
	input  rst,
  // Input Channel
	input  [BITWIDTH - 1 : 0] ins,
	input  ins_valid,
  output ins_ready,
  // Output Channel
  output [SIZE * (BITWIDTH) - 1 : 0] outs,
  output [SIZE - 1 : 0] outs_valid,
	input  [SIZE - 1 : 0] outs_ready
);

  fork_dataless #(
    .SIZE(SIZE)
  ) control (
    .clk        (clk        ),
    .rst        (rst        ),
    .ins_valid  (ins_valid  ),
    .ins_ready  (ins_ready  ),
    .outs_valid (outs_valid ),
    .outs_ready (outs_ready )
  );

  // Broadcast the input data to all output channels
  assign outs = {SIZE{ins}};

endmodule
