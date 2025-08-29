// shift_reg_break_dv : shift_reg_break_dv({'data_type': 32, 'num_slots': 2})

`timescale 1ns/1ps

`timescale 1ns/1ps
// dataless shift_reg_break_dv Module

module shift_reg_break_dv_dataless #(
  parameter integer NUM_SLOTS = 2
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
      valid_reg <= {NUM_SLOTS{1'b0}};
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

// shift_reg_break_dv Module

module shift_reg_break_dv #(
  parameter integer NUM_SLOTS = 2,
  parameter integer DATA_TYPE = 32
)(
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

  // Internal signals
  wire regEn, inputReady;
  reg [DATA_TYPE - 1 : 0] Memory [0 : NUM_SLOTS - 1];
  
  // Instance of shift_reg_break_dv_dataless to manage handshaking
  shift_reg_break_dv_dataless #(
    .NUM_SLOTS(NUM_SLOTS)
  ) control (
    .clk        (clk        ),
    .rst        (rst        ),
    .ins_valid  (ins_valid  ),
    .ins_ready  (inputReady ),
    .outs_valid (outs_valid ),
    .outs_ready (outs_ready )
  );
  
  // See 'docs/Specs/Buffering/Buffering.md'
  // All the slots share a single handshake control and thus 
  // accept or stall inputs together.
  integer i;
  always @(posedge clk) begin
    if (regEn) begin
      for (i = NUM_SLOTS - 1; i > 0; i = i - 1) begin
        Memory[i] <= Memory[i - 1];
      end
      Memory[0] <= ins;
    end
  end
  
  assign regEn     = inputReady;
  assign ins_ready = inputReady;
  assign outs      = Memory[NUM_SLOTS - 1];

endmodule

