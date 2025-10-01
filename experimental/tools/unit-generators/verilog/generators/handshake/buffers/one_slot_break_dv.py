def generate_one_slot_break_dv(name, params):
  bitwidth = params["bitwidth"]

  if(bitwidth == 0):
    return generate_dataless_one_slot_break_dv(name, {})

  header = "`timescale 1ns/1ps\n"

  one_slot_break_dv_dataless_name = name + "_dataless"
  one_slot_break_dv_dataless = generate_dataless_one_slot_break_dv(one_slot_break_dv_dataless_name, {})

  one_slot_break_dv_body = f"""
// Module of one_slot_break_dv
module {name}(
  input  clk,
  input  rst,
  // Input channel
  input  [{bitwidth} - 1 : 0] ins,
  input  ins_valid,
  output ins_ready,
  // Output channel
  output [{bitwidth} - 1 : 0] outs,
  output outs_valid,
  input  outs_ready
);
  wire regEn, inputReady;
  reg [{bitwidth} - 1 : 0] dataReg = 0;
  
  // Instance of one_slot_break_dv_dataless to manage handshaking
  {one_slot_break_dv_dataless_name} control (
    .clk        (clk       ),
    .rst        (rst       ),
    .ins_valid  (ins_valid ),
    .ins_ready  (inputReady),
    .outs_valid (outs_valid),
    .outs_ready (outs_ready)
  );

  always @(posedge clk) begin
    if (rst) begin
      dataReg <= {{{bitwidth}{{1'b0}}}};
    end else if (regEn) begin
      dataReg <= ins;
    end
  end

  assign ins_ready = inputReady;
  assign regEn = inputReady & ins_valid;
  assign outs = dataReg;

endmodule

"""

  return header + one_slot_break_dv_dataless + one_slot_break_dv_body

def generate_dataless_one_slot_break_dv(name, params):
  return f"""
  `timescale 1ns/1ps

// Module of dataless_one_slot_break_dv
module {name} (
  input  clk,
  input  rst,
  // Input channel
  input  ins_valid,
  output ins_ready,
  // Output channel
  output outs_valid,
  input  outs_ready
);
  // Define internal signals
  reg outputValid = 0;

  always @(posedge clk) begin
    if (rst) begin
      outputValid <= 0;
    end else begin
      outputValid <= ins_valid | (~outs_ready & outputValid);
    end
  end

  assign ins_ready = ~outputValid | outs_ready;
  assign outs_valid = outputValid;
  
endmodule
"""
