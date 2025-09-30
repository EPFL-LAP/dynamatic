def generate_shift_reg_break_dv(name, params):

    num_slots = params["num_slots"]
    data_type = params["data_type"]

    header = "`timescale 1ns/1ps\n"

    dataless_shift_reg_break_dr_name = "shift_reg_break_dv_dataless"
    dataless_shift_reg_break_dvr = generate_dataless_shift_reg_break_dv(dataless_shift_reg_break_dr_name, {"num_slots": num_slots})

    shift_reg_break_dvr_body = f"""
// Module of shift_reg_break_dv

module {name}(
  input  clk,
  input  rst,
  // Input channel
  input  [{data_type} - 1 : 0] ins,
  input  ins_valid,
  output ins_ready,
  // Output channel
  output [{data_type} - 1 : 0] outs,
  output outs_valid,
  input  outs_ready
);

  // Internal signals
  wire regEn, inputReady;
  reg [{data_type} - 1 : 0] Memory [0 : {num_slots} - 1];
  
  // Instance of shift_reg_break_dv_dataless to manage handshaking
  {dataless_shift_reg_break_dr_name} control (
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
      for (i = {num_slots} - 1; i > 0; i = i - 1) begin
        Memory[i] <= Memory[i - 1];
      end
      Memory[0] <= ins;
    end
  end
  
  assign regEn     = inputReady;
  assign ins_ready = inputReady;
  assign outs      = Memory[{num_slots} - 1];

endmodule
"""

    return header + dataless_shift_reg_break_dvr + shift_reg_break_dvr_body

def generate_dataless_shift_reg_break_dv(name, params):

    num_slots = params["num_slots"]

    return f"""
`timescale 1ns/1ps
// Module of dataless_shift_reg_break_dv

module {name}(
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
  reg  [{num_slots}-1:0] valid_reg;
  wire             regEn;

  // See 'docs/Specs/Buffering/Buffering.md'
  // All the slots share a single handshake control and thus 
  // accept or stall inputs together.
  always @(posedge clk) begin
    if (rst) begin
      valid_reg <= {{{num_slots}{{1'b0}}}};
    end else begin
      if (regEn) begin
        valid_reg[{num_slots}-1:1] <= valid_reg[{num_slots}-2:0];
        valid_reg[0]         <= ins_valid;
      end
    end
  end

  assign outs_valid = valid_reg[{num_slots}-1];
  assign regEn      = ~outs_valid | outs_ready;
  assign ins_ready  = regEn;

endmodule
"""