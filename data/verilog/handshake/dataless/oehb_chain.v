`timescale 1ns/1ps
module oehb_dataless_chain #(
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
    for (i = 0; i < NUM_SLOTS; i = i + 1) begin : gen_oehb_dataless_chain
      oehb_dataless oehb_dataless_inst (
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