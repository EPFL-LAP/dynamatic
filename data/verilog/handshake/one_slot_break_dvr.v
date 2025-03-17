`timescale 1ns/1ps
module dvr #(
  parameter DATA_TYPE = 32
) (
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
  wire enable, inputReady;
  reg [DATA_TYPE - 1 : 0] dataReg = 0;
  
  // Instance of dvr_dataless to manage handshaking
  dvr_dataless control (
    .clk        (clk       ),
    .rst        (rst       ),
    .ins_valid  (ins_valid ),
    .ins_ready  (inputReady),
    .outs_valid (outs_valid),
    .outs_ready (outs_ready)
  );

  always @(posedge clk) begin
    if (rst) begin
      dataReg <= {DATA_TYPE{1'b0}};
    end else if (enable) begin
      dataReg <= ins;
    end
  end

  assign ins_ready = inputReady;
  assign enable    = ins_valid & inputReady;
  assign outs      = dataReg;

endmodule