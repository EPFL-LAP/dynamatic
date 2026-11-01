`timescale 1ns/1ps
module init_dataless (
  input  clk,
  input  rst,
  // Input channel
  input  ins_valid,
  output ins_ready,
  // Output channel
  output outs_valid,
  input  outs_ready
);
  // the register starts full, so one token comes out before any input
  reg fullReg = 1;
  wire outputValid;

  assign outputValid = ins_valid | fullReg;

  always @(posedge clk) begin
    if (rst) begin
      fullReg <= 1;
    end else begin
      fullReg <= outputValid & ~outs_ready;
    end
  end

  assign ins_ready = ~fullReg;
  assign outs_valid = outputValid;

endmodule
