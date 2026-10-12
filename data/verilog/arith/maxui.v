`timescale 1ns/1ps
module maxui #(
  parameter DATA_TYPE = 32
)(
  input  clk,
  input  rst,
  input  [DATA_TYPE - 1 : 0] lhs,
  input  lhs_valid,
  input  [DATA_TYPE - 1 : 0] rhs,
  input  rhs_valid,
  input  result_ready,
  output [DATA_TYPE - 1 : 0] result,
  output result_valid,
  output lhs_ready,
  output rhs_ready
);
  join_type #(
    .SIZE(2)
  ) join_inputs (
    .ins_valid  ({rhs_valid, lhs_valid}),
    .outs_ready (result_ready             ),
    .ins_ready  ({rhs_ready, lhs_ready}  ),
    .outs_valid (result_valid             )
  );

  assign result = ($unsigned(lhs) > $unsigned(rhs)) ? lhs : rhs;
endmodule
