`timescale 1ns/1ps
module addf #(
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
  
  // Assert that DATA_TYPE is 32
  //assert(DATA_TYPE == 32) else $error("addf currently only supports 32-bit floating point operands");
  
  wire join_valid, oehb_ready, buff_valid;

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

  fadd_32ns_32ns_32_13_full_dsp_1_ip fadd_32ns_32ns_32_13_full_dsp_1_ip_u (
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