`timescale 1ns/1ps
module merge # (
  parameter SIZE = 2,
  parameter DATA_TYPE = 32
)(
  input  clk,
  input  rst,
  // Input channels
  input  [SIZE * DATA_TYPE - 1 : 0] ins, 
  input  [SIZE - 1 : 0] ins_valid,
  output [SIZE - 1 : 0] ins_ready,
  // Output channel
  output [DATA_TYPE - 1 : 0] outs,
  output outs_valid,
  input  outs_ready
);

  reg tmp_valid_out;
  reg [SIZE - 1 : 0] tmp_ready_out;
  reg [DATA_TYPE - 1 : 0] tmp_data_out;
  integer i;
  integer cnt;

  always @(*) begin
    tmp_valid_out = 0;
    tmp_ready_out = {SIZE{1'b0}}; 
    tmp_data_out = ins[0 +: DATA_TYPE];

    cnt = 1;
    for (i = 0; i < SIZE; i = i + 1) begin
      if (cnt == 1 && ins_valid[i]) begin
        tmp_data_out = ins[i * DATA_TYPE +: DATA_TYPE];
        tmp_valid_out = 1;
        tmp_ready_out[i] = outs_ready;
        cnt = 0;
      end
    end
  end
  
  // The outs channel is not persistent, meaning the data payload 
  // may change while valid remains high
  assign outs = tmp_data_out;
  assign outs_valid = tmp_valid_out;
  assign ins_ready = tmp_ready_out;

endmodule
