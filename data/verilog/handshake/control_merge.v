`timescale 1ns/1ps
  module control_merge #(
    parameter SIZE = 2,
    parameter DATA_TYPE = 1,
    parameter INDEX_TYPE = 1
  )(
    input  clk,
    input  rst,
    // Input Channels
    input  [SIZE * (DATA_TYPE) - 1 : 0] ins,    // Input Channel
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
    //! Specified the length of the vector compared with the original implementation.
    //! Jiantao, 21/07/2024
    wire [INDEX_TYPE - 1 : 0] index_internal;

    // Instantiate control_merge_dataless
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

    // Since no data output for outs, I directly assign 0 to it
    // assign outs = ins[index_internal[0]];
    assign outs = 0;

  endmodule
