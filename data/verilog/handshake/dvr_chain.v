`timescale 1ns/1ps
module dvr_chain #(
  parameter DATA_TYPE = 32,
  parameter NUM_SLOTS = 4
) (
  input  clk,
  input  rst,
  // Input channel
  input  [DATA_TYPE - 1 : 0] ins,
  input  ins_valid,
  output ins_ready,
  // Output channel
  output [DATA_TYPE - 1 : 0] outs,
  output outs_valid,
  input  outs_ready
);

  wire [DATA_TYPE - 1 : 0] data_signals [0 : NUM_SLOTS];
  wire valid_signals [0 : NUM_SLOTS];
  wire ready_signals [0 : NUM_SLOTS];

  assign data_signals[0] = ins;
  assign valid_signals[0] = ins_valid;
  assign ins_ready = ready_signals[0];

  assign outs = data_signals[NUM_SLOTS];
  assign outs_valid = valid_signals[NUM_SLOTS];
  assign ready_signals[NUM_SLOTS] = outs_ready;

  genvar i;
  generate
    for (i = 0; i < NUM_SLOTS; i = i + 1) begin : gen_dvr_chain
      dvr #(
        .DATA_TYPE(DATA_TYPE)
      ) dvr_inst (
        .clk        (clk),
        .rst        (rst),
        .ins        (data_signals[i]),
        .ins_valid  (valid_signals[i]),
        .ins_ready  (ready_signals[i]),
        .outs       (data_signals[i+1]),
        .outs_valid (valid_signals[i+1]),
        .outs_ready (ready_signals[i+1])
      );
    end
  endgenerate

endmodule