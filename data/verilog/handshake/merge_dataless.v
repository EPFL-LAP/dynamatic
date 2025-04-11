`timescale 1ns/1ps
module merge_dataless #(
  parameter SIZE = 2
)(
  input  clk,
  input  rst,
  // Input Channels
  input  [SIZE - 1 : 0] ins_valid,
  output [SIZE - 1 : 0] ins_ready,
  // Output Channel
  output outs_valid,
  input  outs_ready
);

  // Internal signal declaration
  wire tehb_pvalid;
  wire tehb_ready;

  merge_notehb_dataless #(
    .INPUTS(SIZE)
  ) merge_ins (
    .clk        (clk          ),
    .rst        (rst          ),
    .ins_valid  (ins_valid    ),
    .ins_ready  (ins_ready    ),
    .outs_valid (tehb_pvalid),
    .outs_ready (tehb_ready)
  );

  tehb_dataless tehb (
    .clk        (clk         ),
    .rst        (rst         ),
    .ins_valid  (tehb_pvalid ),
    .outs_ready (outs_ready  ),
    .outs_valid (outs_valid  ),
    .ins_ready  (tehb_ready  )
  );

endmodule
