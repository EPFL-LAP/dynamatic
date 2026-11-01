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

  // the merge's output goes through a one slot break R buffer (tehb),
  // like the VHDL merge
  wire tehb_ready;

  always @(*) begin
    tmp_valid_out = 0;
    tmp_ready_out = {SIZE{1'b0}};
    for (i = 0; i < SIZE; i = i + 1) begin
      if (ins_valid[i] && !tmp_valid_out) begin
        tmp_valid_out = 1;
        tmp_ready_out[i] = tehb_ready;
      end
    end
  end

  assign ins_ready = tmp_ready_out;

  tehb_dataless tehb (
    .clk        (clk          ),
    .rst        (rst          ),
    .ins_valid  (tmp_valid_out),
    .ins_ready  (tehb_ready   ),
    .outs_valid (outs_valid   ),
    .outs_ready (outs_ready   )
  );

endmodule
