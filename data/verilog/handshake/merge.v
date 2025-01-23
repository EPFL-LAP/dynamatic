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

  merge_notehb #(
    .INPUTS(SIZE),
    .DATA_TYPE(DATA_TYPE)
  ) merge_ins (
    .clk        (clk          ),
    .rst        (rst          ),
    .ins        (ins          ),
    .ins_valid  (ins_valid    ),
    .ins_ready  (ins_ready    ),
    .outs       (tehb_data_in ),
    .outs_valid (tehb_pvalid  ),
    .outs_ready (tehb_ready   )
  );

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
