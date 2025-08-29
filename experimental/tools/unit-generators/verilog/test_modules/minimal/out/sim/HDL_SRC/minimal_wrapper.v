module minimal_wrapper(
  input [31:0] x,
  input  x_valid,
  input  start_valid,
  input  clk,
  input  rst,
  input  out0_ready,
  input  end_ready,
  output  x_ready,
  output  start_ready,
  output [31:0] out0,
  output  out0_valid,
  output  end_valid
);
  wire [31:0] minimal_wrapped_out0;
  wire  minimal_wrapped_out0_valid;
  wire  minimal_wrapped_out0_ready;
  wire  minimal_wrapped_end_valid;
  wire  minimal_wrapped_end_ready;

  assign out0 = minimal_wrapped_out0;
  assign out0_valid = minimal_wrapped_out0_valid;
  assign minimal_wrapped_out0_ready = out0_ready;
  assign end_valid = minimal_wrapped_end_valid;
  assign minimal_wrapped_end_ready = end_ready;

  minimal minimal_wrapped(
    .x (x),
    .x_valid (x_valid),
    .x_ready (x_ready),
    .start_valid (start_valid),
    .start_ready (start_ready),
    .clk (clk),
    .rst (rst),
    .out0 (minimal_wrapped_out0),
    .out0_valid (minimal_wrapped_out0_valid),
    .out0_ready (minimal_wrapped_out0_ready),
    .end_valid (minimal_wrapped_end_valid),
    .end_ready (minimal_wrapped_end_ready)
  );

endmodule
