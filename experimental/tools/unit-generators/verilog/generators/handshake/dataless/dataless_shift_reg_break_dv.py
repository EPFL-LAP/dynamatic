def generate_dataless_shift_reg_break_dv(name, params):

    num_slots = params["num_slots"]

    return f"""
`timescale 1ns/1ps
// Module of dataless_shift_reg_break_dv

module {name} #(
  parameter integer NUM_SLOTS = {num_slots}
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
  reg  [NUM_SLOTS-1:0] valid_reg;
  wire             regEn;

  // See 'docs/Specs/Buffering/Buffering.md'
  // All the slots share a single handshake control and thus 
  // accept or stall inputs together.
  always @(posedge clk) begin
    if (rst) begin
      valid_reg <= {{NUM_SLOTS{{1'b0}}}};
    end else begin
      if (regEn) begin
        valid_reg[NUM_SLOTS-1:1] <= valid_reg[NUM_SLOTS-2:0];
        valid_reg[0]         <= ins_valid;
      end
    end
  end

  assign outs_valid = valid_reg[NUM_SLOTS-1];
  assign regEn      = ~outs_valid | outs_ready;
  assign ins_ready  = regEn;

endmodule
"""