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
  wire [DATA_TYPE - 1 : 0] tehb_data_in;
  wire tehb_pvalid;
  wire tehb_ready;

  reg [DATA_TYPE - 1 : 0] tmp_data_out;
  reg tmp_valid_out = 0;
  integer i;

  always @(*) begin
    tmp_valid_out = 0;
    tmp_data_out = ins[0 +: DATA_TYPE];
    for (i = SIZE - 1; i >= 0; i = i - 1) begin
      if (ins_valid[i]) begin
        tmp_data_out = ins[i * DATA_TYPE +: DATA_TYPE];
        tmp_valid_out = 1;
      end
    end
  end

  assign tehb_data_in = tmp_data_out;
  assign tehb_pvalid = tmp_valid_out;
  assign ins_ready = {SIZE{tehb_ready}};

  tehb #(
    .DATA_TYPE(DATA_TYPE)
  ) tehb_inst (
    .clk        (clk         ),
    .rst        (rst         ),
    .ins_valid  (tehb_pvalid ),
    .outs_ready (outs_ready  ),
    .outs_valid (outs_valid  ),
    .ins_ready  (tehb_ready  ),
    .ins        (tehb_data_in),
    .outs       (outs        )
  );

endmodule
