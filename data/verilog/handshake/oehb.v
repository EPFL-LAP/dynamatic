`timescale 1ns/1ps
module oehb #(
  parameter DATA_TYPE = 32
) (
  input  clk,
  input  rst,
  // Input channel
  input  [DATA_TYPE - 1 : 0] ins,
  input  ins_valid,
  output ins_ready,
  // Output channel
  output reg [DATA_TYPE - 1 : 0] outs,
  output outs_valid,
  input  outs_ready
);
  wire regEn, inputReady;
  
  // Instance of oehb_dataless to manage handshaking
  oehb_dataless control (
    .clk        (clk       ),
    .rst        (rst       ),
    .ins_valid  (ins_valid ),
    .ins_ready  (inputReady),
    .outs_valid (outs_valid),
    .outs_ready (outs_ready)
  );

  always @(posedge clk, posedge rst) begin
    if (rst) begin
      outs <= {DATA_TYPE{1'b0}};
    end else if (regEn) begin
      outs <= ins;
    end
  end

  assign ins_ready = inputReady;
  assign regEn = inputReady & ins_valid;

endmodule
