`timescale 1ns/1ps

module dvse_dataless #(
  parameter integer DEPTH = 4
)(
  input  clk,
  input  rst,
  // Inputs
  input  ins_valid,
  input  outs_ready,
  // Outputs
  output outs_valid,
  output ins_ready
);

  // Internal signals
  reg  [DEPTH-1:0] valid_reg;
  wire             regEn;

  always @(posedge clk) begin
    if (rst) begin
      valid_reg <= {DEPTH{1'b0}};
    end else begin
      if (regEn) begin
        valid_reg[DEPTH-1:1] <= valid_reg[DEPTH-2:0];
        valid_reg[0]         <= ins_valid;
      end
    end
  end

  assign outs_valid = valid_reg[DEPTH-1];
  assign regEn      = ~outs_valid | outs_ready;
  assign ins_ready  = regEn;

endmodule