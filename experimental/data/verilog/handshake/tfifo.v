module tfifo #(
  parameter SIZE = 2,
  parameter BITWIDTH = 32
)(
  input  clk,
  input  rst,
  input  [BITWIDTH - 1 : 0] ins,
  input  ins_valid,
  input  outs_ready,
  output [BITWIDTH - 1 : 0] outs,
  output outs_valid,
  output ins_ready
);
  wire mux_sel;
  wire fifo_valid, fifo_ready;
  wire fifo_pvalid, fifo_nready;
  wire [BITWIDTH - 1 : 0] fifo_in, fifo_out;

  // Dataout assignment
  assign outs = mux_sel ? fifo_out : ins;

  assign outs_valid = ins_valid || fifo_valid;
  assign ins_ready = fifo_ready || outs_ready;
  assign fifo_pvalid = ins_valid && (!outs_ready || fifo_valid);
  assign mux_sel = fifo_valid;

  assign fifo_nready = outs_ready;
  assign fifo_in = ins;

  elastic_fifo_inner #(
    .SIZE     (SIZE    ), 
    .BITWIDTH (BITWIDTH)
  ) fifo (
    .clk        (clk        ),
    .rst        (rst        ),
    .ins        (fifo_in    ),
    .ins_valid  (fifo_pvalid),
    .outs_ready (fifo_nready),
    .outs       (fifo_out   ),
    .outs_valid (fifo_valid ),
    .ins_ready  (fifo_ready )
  );


endmodule