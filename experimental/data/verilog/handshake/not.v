module logic_not #(
  parameter DATA_WIDTH = 32
)(
  input  clk,
  input  rst,
  // Input channel
  input  [DATA_WIDTH - 1 : 0] ins,
  input  ins_valid,
  output ins_ready,
  // Output channel
  output [DATA_WIDTH - 1 : 0] outs,
  output outs_valid,
  input  outs_ready
);
  assign outs = ~ins;
  assign outs_valid = ins_valid;
  assign ins_ready = outs_ready;

endmodule
