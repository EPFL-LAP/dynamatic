module divui #(
  parameter BITWIDTH = 32
)(
  // inputs
  input  clk,
  input  rst,
  input  [BITWIDTH - 1 : 0] lhs,
  input  lhs_valid,
  input  [BITWIDTH - 1 : 0] rhs,
  input  rhs_valid,
  input  result_ready,
  // outputs
  output [BITWIDTH - 1 : 0] result,
  output result_valid,
  output lhs_ready,
  output rhs_ready
);

  wire join_valid;

  // Instantiate the join node
  join_type #(
    .SIZE(2)
  ) join_inputs (
    .ins_valid  ({rhs_valid, lhs_valid}),
    .outs_ready (result_ready             ),
    .ins_ready  ({rhs_ready, lhs_ready}  ),
    .outs_valid (join_valid             )
  );

  array_RAM_udiv_32ns_32ns_32_36_1 #(
    .ID(1),
    .NUM_STAGE(36),
    .din0_WIDTH(32),
    .din1_WIDTH(32),
    .dout_WIDTH(32)
  ) array_RAM_udiv_32ns_32ns_32_36_1_U1 (
    .clk(clk),
    .reset(rst),
    .ce(result_ready),
    .din0(lhs),
    .din1(rhs),
    .dout(result)
  );

  delay_buffer #(
    .SIZE(35)
  ) buff (
    .clk(clk),
    .rst(rst),
    .valid_in(join_valid),
    .ready_in(result_ready),
    .valid_out(result_valid)
  );


endmodule