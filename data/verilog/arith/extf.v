`timescale 1ns/1ps
module extf #(
  parameter INPUT_TYPE = 32,
  parameter OUTPUT_TYPE = 64
)(
  // inputs
  input  clk,
  input  rst,
  input  [INPUT_TYPE - 1 : 0] ins,
  input  ins_valid,
  output  ins_ready,
  // outputs
  output [OUTPUT_TYPE - 1 : 0] outs,
  output outs_valid,
  input outs_ready
);

  __xls_float_ips__extf32_0_next ip (
    .clk(clk),
    .rst(rst),
    .xls_float_ips__ins(ins),
    .xls_float_ips__ins_vld(ins_valid),
    .xls_float_ips__ins_rdy(ins_ready),
    .xls_float_ips__outs(outs),
    .xls_float_ips__outs_vld(outs_valid),
    .xls_float_ips__outs_rdy(outs_ready)
  );

endmodule
