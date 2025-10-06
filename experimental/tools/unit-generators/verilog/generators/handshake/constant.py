def generate_constant(name, params):
    bitwidth = params["bitwidth"]
    value = params["value"]

    constant = f"""
// Module of constant

module {name}(
  input                       clk,
  input                       rst,
  // Input Channel
  input                       ctrl_valid,
  output                      ctrl_ready,
  // Output Channel
  output [{bitwidth} - 1 : 0] outs,
  output                      outs_valid,
  input                       outs_ready
);
  assign outs       = {bitwidth}'b{value};
  assign outs_valid = ctrl_valid;
  assign ctrl_ready = outs_ready;

endmodule
"""


    return constant
