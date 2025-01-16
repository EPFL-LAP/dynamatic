`timescale 1ns/1ps
module divsi #(
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
  output [DATA_TYPE - 1 : 0] result,
  output result_valid,
  output lhs_ready,
  output rhs_ready
);

  xls_divsi32 ip (
    .clk(clk),
    .rst(rst),
    .lhs(lhs),
    .lhs_valid(lhs_valid),
    .lhs_ready(lhs_ready),
    .rhs(rhs),
    .rhs_valid(rhs_valid),
    .rhs_ready(rhs_ready),
    .result(result),
    .result_valid(result_valid),
    .result_ready(result_ready)
  );

endmodule
