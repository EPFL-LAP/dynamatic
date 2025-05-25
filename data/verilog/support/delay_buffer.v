`timescale 1ns/1ps
module delay_buffer #(
  parameter SIZE = 32
) (
  input  clk,
  input  rst,
  input  valid_in,
  input  ready_in,
  output valid_out
);
  integer i;
  reg [SIZE - 1 : 0] regs = 0;

  always @(posedge clk) begin
    if (rst)
      regs[0] <= 0;
    else if (ready_in)
      regs[0] <= valid_in;
  end

  always @(posedge clk) begin
    if (rst) begin
      for (i = 1; i < SIZE; i = i + 1) begin
        regs[i] <= 0;
      end
    end else if (ready_in) begin
      for (i = 1; i < SIZE; i = i + 1) begin
        regs[i] <= regs[i - 1];
      end
    end
  end

  assign valid_out = regs[SIZE - 1];
endmodule
