`timescale 1ns/1ps
module ofifo #(
  parameter NUM_SLOTS = 2,
  parameter DATA_TYPE = 32
) (
  input  clk,
  input  rst,
  input  [DATA_TYPE - 1 : 0] ins,
  input  ins_valid,
  input  outs_ready,
  output [DATA_TYPE - 1 : 0] outs,
  output outs_valid,
  output ins_ready
);
  wire [DATA_TYPE - 1 : 0] tehb_dataOut, fifo_dataOut;
  wire tehb_valid, tehb_ready;
  wire fifo_valid, fifo_ready;
  
  tehb #(
    .DATA_TYPE(DATA_TYPE)
  ) tehb_module (
    .clk        (clk         ),
    .rst        (rst         ),
    .ins        (ins         ),
    .ins_valid  (ins_valid   ),
    .outs_ready (fifo_ready  ),
    .outs       (tehb_dataOut),
    .outs_valid (tehb_valid  ),
    .ins_ready  (tehb_ready  )
  );

  elastic_fifo_inner #(
    .NUM_SLOTS  (NUM_SLOTS ),
    .DATA_TYPE (DATA_TYPE)
  ) fifo (
    .clk        (clk         ),
    .rst        (rst         ),
    .ins        (tehb_dataOut),
    .ins_valid  (tehb_valid  ),
    .outs_ready (outs_ready  ),
    .outs       (fifo_dataOut),
    .outs_valid (fifo_valid  ),
    .ins_ready  (fifo_ready  )
  );

  assign outs = fifo_dataOut;
  assign outs_valid = fifo_valid;
  assign ins_ready = tehb_ready;

endmodule