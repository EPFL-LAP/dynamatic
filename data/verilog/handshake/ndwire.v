`timescale 1ns/1ps
module ndwire #(
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

  ndwire_dataless #(
    .SIZE(DATA_TYPE)
  ) control (
    .clk        (clk        ),
    .rst        (rst        ),
    .ins_valid  (ins_valid  ),
    .ins_ready  (ins_ready  ),
    .outs_valid (outs_valid ),
    .outs_ready (outs_ready )
  );
  
  assign outs = ins;

endmodule
