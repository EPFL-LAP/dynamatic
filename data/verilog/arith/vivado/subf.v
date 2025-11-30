`timescale 1ns/1ps
module subf #(
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

  //assert(DATA_TYPE == 32) else $fatal("subf currently only supports 32-bit operands");

  wire join_valid, oehb_ready, buff_valid;

  // subf is the same as addf, but we flip the sign bit of rhs
  wire [DATA_TYPE - 1 : 0] rhs_neg;

  // intermediate input signals for IEEE-754 to Flopoco-simple-float conversion
  wire [ DATA_TYPE + 1 :0] ip_lhs, ip_rhs;

  // intermediate output signal for Flopoco-simple-float to IEEE-754 conversion
  wire [ DATA_TYPE + 1 :0] ip_result;

  // Instantiate the join node
  join_type #(
    .SIZE(2)
  ) join_inputs (
    .ins_valid  ({rhs_valid, lhs_valid}),
    .outs_ready (oehb_ready             ),
    .ins_ready  ({rhs_ready, lhs_ready}  ),
    .outs_valid (join_valid             )
  );

  oehb_dataless oehb_lhs (
    .clk(clk),
    .rst(rst),
    .ins_valid(join_valid),
    .ins_ready(oehb_ready),
    .outs_valid(buff_valid),
    .outs_ready(result_ready)
  ); 

  subf_vitis_hls_single_precision_lat_8 subf_vitis_hls_single_precision_lat_8_u (
    .aclk                 ( clk ),
    .aclken               ( 1'b1 ),
    .s_axis_a_tvalid      ( join_valid ),
    .s_axis_a_tdata       ( lhs ),
    .s_axis_b_tvalid      ( join_valid ),
    .s_axis_b_tdata       ( rhs ),
    .m_axis_result_tvalid ( result_valid ),
    .m_axis_result_tdata  ( result )
  );

endmodule