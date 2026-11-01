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

  // the merge's output goes through a one slot break R buffer (tehb),
  // like the VHDL merge
  wire tehb_ready;

  always @(*) begin
    tmp_valid_out = 0;
    tmp_ready_out = {SIZE{1'b0}};
    tmp_data_out = ins[0 +: DATA_TYPE];

    cnt = 1;
    for (i = 0; i < SIZE; i = i + 1) begin
      if (cnt == 1 && ins_valid[i]) begin
        tmp_data_out = ins[i * DATA_TYPE +: DATA_TYPE];
        tmp_valid_out = 1;
        tmp_ready_out[i] = tehb_ready;
        cnt = 0;
      end
    end
  end

  assign ins_ready = tmp_ready_out;

  tehb #(
    .DATA_TYPE(DATA_TYPE)
  ) tehb (
    .clk        (clk          ),
    .rst        (rst          ),
    .ins        (tmp_data_out ),
    .ins_valid  (tmp_valid_out),
    .ins_ready  (tehb_ready   ),
    .outs       (outs         ),
    .outs_valid (outs_valid   ),
    .outs_ready (outs_ready   )
  );

endmodule
