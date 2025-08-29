module minimal(
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

  assign out0 = x;
  assign out0_valid = x_valid;
  assign x_ready = out0_ready;
  assign end_valid = start_valid;
  assign start_ready = end_ready;

endmodule
