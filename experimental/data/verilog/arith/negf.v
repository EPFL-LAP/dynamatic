module negf #(
  parameter BITWIDTH = 32
)(
  // inputs
  input  clk,
  input  rst,
  input  [BITWIDTH - 1 : 0] ins,
  input  ins_valid,
  input  ins_ready,
  // outputs
  output [BITWIDTH - 1 : 0] outs,
  output outs_valid,
  output outs_ready
);

  assign outs[BITWIDTH-1] = ins[BITWIDTH-1] ^ 1'b1;
  assign outs[BITWIDTH-2:0] = ins[BITWIDTH-2:0];
  assign outs_valid = ins_valid;
  assign ins_ready = outs_ready;

endmodule