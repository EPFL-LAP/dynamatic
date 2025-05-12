`timescale 1ns/1ps
  module control_merge #(
    parameter SIZE = 2,
    parameter DATA_TYPE = 32,
    parameter INDEX_TYPE = 1
  )(
    input  clk,
    input  rst,
    // Input Channels
    input  [SIZE * (DATA_TYPE) - 1 : 0] ins,
    input  [SIZE - 1 : 0] ins_valid,
    output [SIZE - 1 : 0] ins_ready,
    // Data Output Channel
    output [DATA_TYPE - 1 : 0] outs,
    output outs_valid,
    input  outs_ready,
    // Index Output Channel
    output [INDEX_TYPE - 1 : 0] index,
    output index_valid,
    input  index_ready
  );
    wire [INDEX_TYPE - 1 : 0] index_internal;

    control_merge_dataless #(
      .SIZE(SIZE),
      .INDEX_TYPE(INDEX_TYPE)
    ) control (
      .clk          (clk            ),
      .rst          (rst            ),
      .ins_valid    (ins_valid      ),
      .ins_ready    (ins_ready      ),
      .outs_valid   (outs_valid     ),
      .outs_ready   (outs_ready     ),
      .index        (index_internal ),
      .index_valid  (index_valid    ),
      .index_ready  (index_ready    )
    );

    assign index = index_internal;

    assign outs = ins[index_internal * DATA_TYPE +: DATA_TYPE];
  endmodule
