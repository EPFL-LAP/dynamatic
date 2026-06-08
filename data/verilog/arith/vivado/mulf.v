`timescale 1ns/1ps
module mulf #(
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

  //assert(DATA_TYPE == 32) else $fatal("mulf currently only supports 32-bit inputs");

  wire join_valid;
  wire buff_valid, oehb_ready;
  wire [ DATA_TYPE - 1 :0] tmp_result;

  // Instantiate the join node
  join_type #(
    .SIZE(2)
  ) join_inputs (
    .ins_valid  ({rhs_valid, lhs_valid}),
    .outs_ready (oehb_ready             ),
    .ins_ready  ({rhs_ready, lhs_ready}  ),
    .outs_valid (join_valid             )
  );

  // Accept only inputs both of 32-bit floating point format
  if (DATA_TYPE != 32) begin
    initial begin
      $fatal("mulf currently only supports 32-bit inputs");
    end
  end

  oehb #(
    .DATA_TYPE(DATA_TYPE)
  ) oehb_lhs (
    .clk(clk),
    .rst(rst),
    .ins(tmp_result),
    .ins_valid(buff_valid),
    .ins_ready(oehb_ready),
    .outs(result),
    .outs_valid(result_valid),
    .outs_ready(result_ready)
  );


  //------------------------Instantiation------------------
  mulf_vitis_hls_single_precision_lat_4 mulf_vitis_hls_single_precision_lat_4_u (
    .aclk                 ( clk ),
    .aclken               ( oehb_ready ),
    .s_axis_a_tvalid      ( join_valid ),
    .s_axis_a_tdata       ( lhs ),
    .s_axis_b_tvalid      ( join_valid ),
    .s_axis_b_tdata       ( rhs ),
    .m_axis_result_tvalid ( buff_valid ),
    .m_axis_result_tdata  ( tmp_result )
);

endmodule