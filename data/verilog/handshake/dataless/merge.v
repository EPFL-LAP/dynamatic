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

  reg tmp_valid_out;
  reg [SIZE - 1 : 0] tmp_ready_out;
  integer i;

  always @(*) begin
    tmp_valid_out = 0;
    tmp_ready_out = {SIZE{1'b0}}; 
    for (i = 0; i < SIZE; i = i + 1) begin
      if (ins_valid[i] && !tmp_valid_out) begin
        tmp_valid_out = 1;
        tmp_ready_out[i] = outs_ready;
      end
    end
  end
  
  assign outs_valid = tmp_valid_out;
  assign ins_ready = tmp_ready_out;

endmodule
