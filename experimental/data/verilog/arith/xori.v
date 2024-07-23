module xori #(
  parameter DATA_WIDTH = 32
)(
  // inputs
  input  clk,
  input  rst,
  input  [DATA_WIDTH - 1 : 0] lhs,
  input  lhs_valid,
  input  [DATA_WIDTH - 1 : 0] rhs,
  input  rhs_valid,
  input  result_ready,
  // outputs
  output [DATA_WIDTH - 1 : 0] result,
  output result_valid,
  output lhs_ready,
  output rhs_ready
);

  // Instantiate the join node
  join_type #(
    .SIZE(2)
  ) join_inputs (
    .ins_valid  ({rhs_valid, lhs_valid}),
    .outs_ready (result_ready             ),
    .ins_ready  ({rhs_ready, lhs_ready}  ),
    .outs_valid (result_valid             )
  );

  assign result = lhs ^ rhs;

endmodule