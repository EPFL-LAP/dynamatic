module ENTITY_NAME #(
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

  wire constant_one = 1'b1;
  wire constant_zero = 1'b0;

  // Instantiate the join node
  join #(
    .SIZE(2)
  ) join_inputs (
    .ins_valid  ({rhs_valid, lhs_valid}),
    .outs_ready (result_ready             ),
    .ins_ready  ({rhs_ready, lhs_ready}  ),
    .outs_valid (result_valid             )
  );

  assign result = constant_one ? (MODIFIER(lhs) COMPARATOR MODIFIER(rhs)) : constant_zero;

endmodule