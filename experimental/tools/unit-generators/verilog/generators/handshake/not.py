def generate_not(name, params):
    bitwidth = params["bitwidth"]

    return f"""
// Module of not
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
  assign outs = ~ins;
  assign outs_valid = ins_valid;
  assign ins_ready = outs_ready;

endmodule
"""
