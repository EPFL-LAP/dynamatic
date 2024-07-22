module end_sync_memless #(
  parameter BITWIDTH = 32;
)(
  input  clk,
  input  rst,
  // Input Channel
  input  [BITWIDTH - 1 : 0] ins,
  input  ins_valid,
  output ins_ready,
  // Output Channel
  output [BITWIDTH - 1 : 0] outs,
  output outs_valid,
  input  outs_ready
);
  end_sync_memless_dataless control (
    .clk        (clk        ),
    .rst        (rst        ),
    .ins_valid  (ins_valid  ),
    .ins_ready  (ins_ready  ),
    .outs_valid (outs_valid ),
    .outs_ready (outs_ready )
  );

  assign outs = ins;

endmodule
