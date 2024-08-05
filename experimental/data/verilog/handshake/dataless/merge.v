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
  reg tehb_pvalid;
  wire tehb_ready;
  integer i;

  always @(*) begin
    tehb_pvalid = 0;
    for (i = SIZE - 1; i >= 0; i = i - 1) begin
      if (ins_valid[i]) begin
        tehb_pvalid = 1;
      end
    end
  end

  // Handling input readiness
  assign ins_ready = {SIZE{tehb_ready}};

  tehb_dataless tehb (
    .clk        (clk         ),
    .rst        (rst         ),
    .ins_valid  (tehb_pvalid ),
    .outs_ready (outs_ready  ),
    .outs_valid (outs_valid  ),
    .ins_ready  (tehb_ready  )
  );

endmodule
