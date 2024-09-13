`timescale 1ns/1ps
module ENTITY_NAME #(
  parameter DATA_TYPE = 32
)(
  // inputs
  input  clk,
  input  rst,
  input  [DATA_TYPE - 1 : 0] lhs,
  input  lhs_valid,
  input  [DATA_TYPE - 1 : 0] rhs,
  input  rhs_valid,
  input  result_ready,
  // outputs
  output result,
  output result_valid,
  output lhs_ready,
  output rhs_ready
);

  wire constant_one = 1'b1;
  wire constant_zero = 1'b0;

  // Instantiate the join node
  join_type #(
    .SIZE(2)
  ) join_inputs (
    .ins_valid  ({rhs_valid, lhs_valid}),
    .outs_ready (result_ready             ),
    .ins_ready  ({rhs_ready, lhs_ready}  ),
    .outs_valid (result_valid             )
  );

  assign result = (MODIFIER(lhs) COMPARATOR MODIFIER(rhs)) ? constant_one : constant_zero;

endmodule