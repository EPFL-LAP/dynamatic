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
    initial begin
      $fatal("control_merge implementation with data signal has a bug. Use beta backend instead");
    end
  endmodule
