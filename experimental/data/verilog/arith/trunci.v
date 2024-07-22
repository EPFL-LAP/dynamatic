module trunci #(
  parameter INPUT_WIDTH = 32,
  parameter OUTPUT_WIDTH = 32
)(
  // inputs
  input  clk,
  input  rst,
  input  [INPUT_WIDTH - 1 : 0] ins,
  input  ins_valid,
  input  ins_ready,
  // outputs
  output [OUTPUT_WIDTH - 1 : 0] outs,
  output outs_valid,
  output outs_ready
);

  assign outs = ins[OUTPUT_WIDTH - 1 : 0];
  assign outs_valid = ins_valid;
  assign ins_ready = outs_ready;

endmodule