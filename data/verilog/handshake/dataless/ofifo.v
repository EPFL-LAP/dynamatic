`timescale 1ns/1ps
module ofifo_dataless #(
  parameter NUM_SLOTS = 2
)(
  input  clk,
  input  rst,
  input  ins_valid,
  input  outs_ready,
  output ins_ready,
  output outs_valid
);
  wire tehb_valid, tehb_ready;
  wire fifo_valid, fifo_ready;

  // Instantiate tehb_dataless
  tehb_dataless tehb (
    .clk        (clk        ),
    .rst        (rst        ),
    .ins_valid  (ins_valid  ),
    .outs_ready (fifo_ready ),
    .outs_valid (tehb_valid ),
    .ins_ready  (tehb_ready )
  );

  elastic_fifo_inner_dataless #(
    .NUM_SLOTS(NUM_SLOTS)
  ) fifo(
    .clk        (clk       ),
    .rst        (rst       ),
    .ins_valid  (tehb_valid),
    .outs_ready (outs_ready),
    .outs_valid (fifo_valid),
    .ins_ready  (fifo_ready)
  );

  assign outs_valid = fifo_valid;
  assign ins_ready = tehb_ready;

endmodule
