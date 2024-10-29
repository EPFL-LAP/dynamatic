`timescale 1ns/1ps

module dvse #(
  parameter integer NUM_SLOTS = 4,
  parameter integer DATA_TYPE = 32
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

  // Internal signals
  wire regEn, inputReady;
  reg [DATA_TYPE - 1 : 0] Memory [0 : NUM_SLOTS - 1];
  
  // Instance of dvse_dataless to manage handshaking
  dvse_dataless #(
    .NUM_SLOTS(NUM_SLOTS)
  ) control (
    .clk        (clk        ),
    .rst        (rst        ),
    .ins_valid  (ins_valid  ),
    .ins_ready  (inputReady ),
    .outs_valid (outs_valid ),
    .outs_ready (outs_ready )
  );
  
  integer i;
  always @(posedge clk) begin
    if (regEn) begin
      for (i = NUM_SLOTS - 1; i > 0; i = i - 1) begin
        Memory[i] <= Memory[i - 1];
      end
      Memory[0] <= ins;
    end
  end
  
  assign regEn     = inputReady;
  assign ins_ready = inputReady;
  assign outs      = Memory[NUM_SLOTS - 1];

endmodule