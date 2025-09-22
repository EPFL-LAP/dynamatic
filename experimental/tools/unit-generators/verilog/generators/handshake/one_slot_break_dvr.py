from generators.handshake.dataless.dataless_one_slot_break_dvr import generate_dataless_one_slot_break_dvr
def generate_one_slot_break_dvr(name, params):

    data_type = params["data_type"]

    header = "`timescale 1ns/1ps\n"

    dataless_one_slot_break_dvr_name = "one_slot_break_dvr_dataless"
    dataless_one_slot_break_dvr = generate_dataless_one_slot_break_dvr(dataless_one_slot_break_dvr_name, {})

    one_slot_break_dvr_body = f"""
// Module of one_slot_break_dvr
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
  wire enable, inputReady;
  reg [{data_type} - 1 : 0] dataReg = 0;
  
  // Instance of one_slot_break_dvr_dataless to manage handshaking
  {dataless_one_slot_break_dvr_name} control (
    .clk        (clk       ),
    .rst        (rst       ),
    .ins_valid  (ins_valid ),
    .ins_ready  (inputReady),
    .outs_valid (outs_valid),
    .outs_ready (outs_ready)
  );

  always @(posedge clk) begin
    if (rst) begin
      dataReg <= {{{data_type}{{1'b0}}}};
    end else if (enable) begin
      dataReg <= ins;
    end
  end

  assign ins_ready = inputReady;
  assign enable    = ins_valid & inputReady;
  assign outs      = dataReg;

endmodule
"""

    return header + dataless_one_slot_break_dvr + one_slot_break_dvr_body