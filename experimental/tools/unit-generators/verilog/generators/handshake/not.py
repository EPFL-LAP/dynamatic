def generate_not(name, params):
    bitwidth = params["bitwidth"]

    verilognot = f"""
`timescale 1ns/1ps
module {name} #(
  parameter DATA_TYPE = {bitwidth}
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
  assign outs = ~ins;
  assign outs_valid = ins_valid;
  assign ins_ready = outs_ready;

endmodule
"""

    return verilognot
