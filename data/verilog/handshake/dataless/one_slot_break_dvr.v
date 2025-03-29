`timescale 1ns/1ps
module one_slot_break_dvr_dataless (
  input  clk,
  input  rst,
  // Input channel
  input  ins_valid,
  output ins_ready,
  // Output channel
  output outs_valid,
  input  outs_ready
);

  // Define internal signals
  reg outputValid = 0;
  reg inputReady = 1;
  wire enable;
  wire stop;

  // Sequential logic for inputReady
  always @(posedge clk) begin
    if (rst) begin
      inputReady <= 1;
    end else begin
      inputReady <= (~stop) & (~enable);
    end
  end

  // Sequential logic for outputValid
  always @(posedge clk) begin
    if (rst) begin
      outputValid <= 0;
    end else begin
      outputValid <= enable | stop;
    end
  end

  // Combinational logic
  assign enable = ins_valid & inputReady;
  assign stop   = outputValid & (~outs_ready);
  // Output assignments
  assign ins_ready  = inputReady;
  assign outs_valid = outputValid;

endmodule