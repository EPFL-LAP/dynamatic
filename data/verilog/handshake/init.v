`timescale 1ns/1ps
module init #(
  parameter DATA_TYPE = 32,
  parameter INITIAL_VALUE = 0
)(
  input  clk,
  input  rst,
  // Input channel
  input  [DATA_TYPE - 1 : 0] ins,
  input  ins_valid,
  output ins_ready,
  // Output channel
  output [DATA_TYPE - 1 : 0] outs,
  output outs_valid,
  input  outs_ready
);
  wire regNotFull;
  wire regEnable;
  reg [DATA_TYPE - 1 : 0] dataReg = INITIAL_VALUE;

  assign regEnable = regNotFull & ins_valid & ~outs_ready;

  init_dataless control (
    .clk        (clk        ),
    .rst        (rst        ),
    .ins_valid  (ins_valid  ),
    .ins_ready  (regNotFull ),
    .outs_valid (outs_valid ),
    .outs_ready (outs_ready )
  );

  always @(posedge clk) begin
    if (rst) begin
      dataReg <= INITIAL_VALUE;
    end else if (regEnable) begin
      dataReg <= ins;
    end
  end

  assign outs = regNotFull ? ins : dataReg;
  assign ins_ready = regNotFull;

endmodule
