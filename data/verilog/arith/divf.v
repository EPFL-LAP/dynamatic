`timescale 1ns/1ps
module divf #(
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

  __xls_float_ips__divf32_0_next ip (
    .clk(clk),
    .rst(rst),
    .xls_float_ips__lhs(lhs),
    .xls_float_ips__lhs_vld(lhs_valid),
    .xls_float_ips__lhs_rdy(lhs_ready),
    .xls_float_ips__rhs(rhs),
    .xls_float_ips__rhs_vld(rhs_valid),
    .xls_float_ips__rhs_rdy(rhs_ready),
    .xls_float_ips__result(result),
    .xls_float_ips__result_vld(result_valid),
    .xls_float_ips__result_rdy(result_ready)
  );

endmodule
