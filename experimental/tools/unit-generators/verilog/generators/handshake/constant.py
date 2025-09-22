def generate_constant(name, params):
    bitwidth = params["bitwidth"]
    value = params["value"]

    verilog_constant = f"""
// Module of constant

`timescale 1ns / 1ps
module {name} #(
  parameter DATA_WIDTH = {bitwidth}
) (
  input                       clk,
  input                       rst,
  // Input Channel
  input                       ctrl_valid,
  output                      ctrl_ready,
  // Output Channel
  output [DATA_WIDTH - 1 : 0] outs,
  output                      outs_valid,
  input                       outs_ready
);
  assign outs       = {bitwidth}'b{value};
  assign outs_valid = ctrl_valid;
  assign ctrl_ready = outs_ready;

endmodule
"""


    return verilog_constant
