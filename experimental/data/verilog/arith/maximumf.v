module maximumf #(
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
  join #(
    .SIZE(2)
  ) join_inputs (
    .ins_valid  ({rhs_valid, lhs_valid}),
    .outs_ready (result_ready             ),
    .ins_ready  ({rhs_ready, lhs_ready}  ),
    .outs_valid (join_valid             )
  );

  my_maxf my_trunc_U1 (
    .ap_clk(clk),
    .ap_rst(rst),
    .a(lhs),
    .b(rhs),
    .ap_return(result)
  );

  delay_buffer #(
    .SIZE(1)
  ) buff (
    .clk(clk),
    .rst(rst),
    .valid_in(join_valid),
    .ready_in(result_ready),
    .valid_out(result_valid)
  );

endmodule