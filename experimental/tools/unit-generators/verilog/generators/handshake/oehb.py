from generators.handshake.dataless.dataless_oehb import generate_dataless_oehb

def generate_oehb(name, params):
  data_type = params["data_type"]

  verilog_header = "`timescale 1ns/1ps\n"

  oehb_dataless_name = name + "_dataless"
  verilog_oehb_dataless = generate_dataless_oehb(oehb_dataless_name, params)

  verilog_oehb_body = f"""
// Module of oehb
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
  wire regEn, inputReady;
  reg [DATA_TYPE - 1 : 0] dataReg = 0;
  
  // Instance of oehb_dataless to manage handshaking
  {oehb_dataless_name} control (
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
    end else if (regEn) begin
      dataReg <= ins;
    end
  end

  assign ins_ready = inputReady;
  assign regEn = inputReady & ins_valid;
  assign outs = dataReg;

endmodule

"""

  return verilog_header + verilog_oehb_dataless + verilog_oehb_body