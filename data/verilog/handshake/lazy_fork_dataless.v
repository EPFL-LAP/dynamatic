`timescale 1ns/1ps
module lazy_fork_dataless #(
  parameter SIZE = 2
)(
  input  clk,
	input  rst,
  // Input Channel
  input  ins_valid,
  output ins_ready,
  // Output Channels
  output reg [SIZE - 1 : 0] outs_valid,
	input  [SIZE - 1: 0] outs_ready
);
  wire allnReady;
  and_n #(
    .SIZE(SIZE)
  ) genericAnd (
    .ins  (outs_ready ),
		.outs (allnReady  )
  );

  // Process to handle output valid signals based on input valid and output readiness
  integer i, j;
  reg [SIZE - 1 : 0] tmp_ready;

  always @(*) begin
    tmp_ready = {SIZE{1'b1}};

    for (i = 0; i < SIZE; i = i + 1) begin
      for (j = 0; j < SIZE; j = j + 1) begin
        if (i != j) begin
          tmp_ready[i] = tmp_ready[i] & outs_ready[j];
        end
      end
    end

    for (i = 0; i < SIZE; i = i + 1) begin
      outs_valid[i] = ins_valid & tmp_ready[i];
    end
  end

  assign ins_ready = allnReady;

endmodule
