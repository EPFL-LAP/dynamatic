`timescale 1ns/1ps
module divf #(
  parameter DATA_WIDTH = 32
)(
  // inputs
  input  clk,
  input  rst,
  input  [DATA_WIDTH - 1 : 0] lhs,
  input  lhs_valid,
  input  [DATA_WIDTH - 1 : 0] rhs,
  input  rhs_valid,
  input  result_ready,
  // outputs
  output [DATA_WIDTH - 1 : 0] result,
  output result_valid,
  output lhs_ready,
  output rhs_ready
);

  //assert(DATA_WIDTH == 32) else $error("divf currently only supports 32-bit floating point operands");

  wire join_valid, oehb_ready, buff_valid;
  wire constant_zero = 1'b0;
  wire open_value;

  // intermediate input signals for IEEE-754 to Flopoco-simple-float conversion
  wire [ DATA_WIDTH + 1 :0] ip_lhs, ip_rhs;

  // intermediate output signal for Flopoco-simple-float to IEEE-754 conversion
  wire [ DATA_WIDTH + 1 :0] ip_result;

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

  oehb #(
    .DATA_WIDTH(1)
  ) oehb_lhs (
    .clk(clk),
    .rst(rst),
    .ins(constant_zero),
    .ins_valid(buff_valid),
    .ins_ready(oehb_ready),
    .outs(open_value),
    .outs_valid(result_valid),
    .outs_ready(result_ready)
  );

  ieee2nfloat_lhs  InputIEEE_32bit (
    .X(lhs),
    .R(ip_lhs)
  );

  ieee2nfloat_rhs  InputIEEE_32bit (
    .X(rhs),
    .R(ip_rhs)
  );

  nfloat2ieee  OutputIEEE_32bit (
    .X(ip_result),
    .R(result)
  );

  ip  FloatingPointDivider (
    .clk(clk),
    .ce(oehb_ready),
    .X(ip_lhs),
    .Y(ip_rhs),
    .R(ip_result)
  );



endmodule