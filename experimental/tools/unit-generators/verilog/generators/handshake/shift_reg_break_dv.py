from generators.handshake.dataless.dataless_shift_reg_break_dv import generate_dataless_shift_reg_break_dv
def generate_shift_reg_break_dv(name, params):

    num_slots = params["num_slots"]
    data_type = params["data_type"]

    verilog_header = "`timescale 1ns/1ps\n"

    dataless_shift_reg_break_dr_name = "shift_reg_break_dv_dataless"
    verilog_dataless_shift_reg_break_dvr = generate_dataless_shift_reg_break_dv(dataless_shift_reg_break_dr_name, params)

    verilog_shift_reg_break_dvr_body = f"""
// Module of shift_reg_break_dv

module {name} #(
  parameter integer NUM_SLOTS = {num_slots},
  parameter integer DATA_TYPE = {data_type}
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
  {dataless_shift_reg_break_dr_name} #(
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
"""

    return verilog_header + verilog_dataless_shift_reg_break_dvr + verilog_shift_reg_break_dvr_body