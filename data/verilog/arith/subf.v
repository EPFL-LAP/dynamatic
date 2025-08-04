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

  delay_buffer #(
    .SIZE(8)
  ) buff (
    .clk(clk),
    .rst(rst),
    .valid_in(join_valid),
    .ready_in(oehb_ready),
    .valid_out(buff_valid)
  );

  oehb_dataless  oehb_lhs (
    .clk(clk),
    .rst(rst),
    .ins_valid(buff_valid),
    .ins_ready(oehb_ready),
    .outs_valid(result_valid),
    .outs_ready(result_ready)
  );

  ieee2nfloat_lhs  InputIEEE_32bit (
    .X(lhs),
    .R(ip_lhs)
  );

  assign rhs_neg = ~rhs + 1;

  ieee2nfloat_rhs  InputIEEE_32bit (
    .X(rhs_neg),
    .R(ip_rhs)
  );

  nfloat2ieee  OutputIEEE_32bit (
    .X(ip_result),
    .R(result)
  );

  ip  FloatingPointAdder (
    .clk(clk),
    .ce(oehb_ready),
    .X(ip_lhs),
    .Y(ip_rhs),
    .R(ip_result)
  );

endmodule