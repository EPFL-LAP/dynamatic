`timescale 1ns/1ps
module tfifo_dataless #(
  parameter NUM_SLOTS = 2
)(
  input  clk,
  input  rst,
  // Input channel
  input  ins_valid,
  output ins_ready,
  // Output channel
  output outs_valid,
  input  outs_ready
);
  wire mux_sel;
  wire fifo_valid, fifo_ready;
  wire fifo_pvalid, fifo_nready;

  assign outs_valid = ins_valid || fifo_valid;
  assign ins_ready = fifo_ready || outs_ready;
  assign fifo_pvalid = ins_valid && (!outs_ready || fifo_valid);
  assign mux_sel = fifo_valid;
  assign fifo_nready = outs_ready;

  elastic_fifo_inner_dataless #(
    .NUM_SLOTS(NUM_SLOTS)
  ) fifo (
    .clk        (clk        ),
    .rst        (rst        ),
    .ins_valid  (fifo_pvalid),
    .outs_ready (fifo_nready),
    .outs_valid (fifo_valid ),
    .ins_ready  (fifo_ready )
  );

endmodule
