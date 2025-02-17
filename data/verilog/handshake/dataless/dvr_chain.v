`timescale 1ns/1ps
module dvr_dataless_chain #(
  parameter NUM_SLOTS = 4
) (
  input  clk,
  input  rst,
  // Input channel
  input  ins_valid,
  output ins_ready,
  // Output channel
  output outs_valid,
  input  outs_ready
);

  wire valid_signals [0 : NUM_SLOTS];
  wire ready_signals [0 : NUM_SLOTS];

  assign valid_signals[0] = ins_valid;
  assign ins_ready = ready_signals[0];

  assign outs_valid = valid_signals[NUM_SLOTS];
  assign ready_signals[NUM_SLOTS] = outs_ready;

  genvar i;
  generate
    for (i = 0; i < NUM_SLOTS; i = i + 1) begin : gen_dvr_dataless_chain
      dvr_dataless dvr_dataless_inst (
        .clk        (clk),
        .rst        (rst),
        .ins_valid  (valid_signals[i]),
        .ins_ready  (ready_signals[i]),
        .outs_valid (valid_signals[i+1]),
        .outs_ready (ready_signals[i+1])
      );
    end
  endgenerate

endmodule