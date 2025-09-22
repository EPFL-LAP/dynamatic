from generators.handshake.dataless.dataless_one_slot_break_dvr import generate_dataless_one_slot_break_dvr
def generate_one_slot_break_dvr(name, params):

    data_type = params["data_type"]

    verilog_header = "`timescale 1ns/1ps\n"

    dataless_one_slot_break_dvr_name = "one_slot_break_dvr_dataless"
    verilog_dataless_one_slot_break_dvr = generate_dataless_one_slot_break_dvr(dataless_one_slot_break_dvr_name, params)

    verilog_one_slot_break_dvr_body = f"""
// Module of one_slot_break_dvr
module {name} #(
  parameter DATA_TYPE = {data_type}
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
  wire enable, inputReady;
  reg [DATA_TYPE - 1 : 0] dataReg = 0;
  
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
      dataReg <= {{DATA_TYPE{{1'b0}}}};
    end else if (enable) begin
      dataReg <= ins;
    end
  end

  assign ins_ready = inputReady;
  assign enable    = ins_valid & inputReady;
  assign outs      = dataReg;

endmodule
"""

    return verilog_header + verilog_dataless_one_slot_break_dvr + verilog_one_slot_break_dvr_body