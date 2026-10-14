`timescale 1ns/1ps
module cosf #(
  parameter DATA_TYPE = 64
)(
  // inputs
  input  clk,
  input  rst,
  input  [DATA_TYPE - 1 : 0] ins,
  input  ins_valid,
  input  outs_ready,
  // outputs
  output [DATA_TYPE - 1 : 0] outs,
  output outs_valid,
  output ins_ready
);

  wire oehb_ready;
  wire buff_valid;

  // The IP core only supports 64-bit operands (see
  // data/vhdl/support/cosf_vitis_hls.vhd)
  cosf_vitis_hls_wrapper ip (
    .clk(clk),
    .reset(rst),
    .ce(oehb_ready),
    .din0(ins),
    .dout(outs)
  );

  oehb_dataless oehb_inst (
    .clk(clk),
    .rst(rst),
    .ins_valid(buff_valid),
    .ins_ready(oehb_ready),
    .outs_valid(outs_valid),
    .outs_ready(outs_ready)
  );

  // The IP core has a latency of 33 cycles: delay_buffer has 33 - 1 cycles of
  // latency, and the OEHB gives the remaining one
  delay_buffer #(
    .SIZE(33 - 1)
  ) buff (
    .clk(clk),
    .rst(rst),
    .valid_in(ins_valid),
    .ready_in(oehb_ready),
    .valid_out(buff_valid)
  );

  assign ins_ready = oehb_ready;

endmodule
