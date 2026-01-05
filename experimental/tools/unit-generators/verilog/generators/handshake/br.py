
def generate_br(name, params):
    bitwidth = params["bitwidth"]
    extra_signals = params.get("extra_signals", None)

    if bitwidth == 0:
        return _generate_br_dataless(name)
    else:
        return _generate_br(name, bitwidth)


def _generate_br(name, bitwidth):

    br_dataless_name = name + "_br_dataless"
    br_dataless = _generate_br_dataless(br_dataless_name)

    body_br = f"""
// Module of br
module {name}(
  input clk,
  input rst,
  // Input Channel
  input [{bitwidth} - 1 : 0] ins,   
  input ins_valid,
  output ins_ready,
  // Output Channel   
  output [{bitwidth} - 1 : 0] outs,
  output outs_valid  ,
  input outs_ready              
);

  {br_dataless_name} control (
    .clk        (clk       ),
    .rst        (rst       ),
    .ins_valid  (ins_valid ),
    .ins_ready  (ins_ready ),
    .outs_valid (outs_valid),
    .outs_ready (outs_ready)
  );
  
  assign outs = ins;

endmodule
"""

    return br_dataless + body_br


def _generate_br_dataless(name):

    body_br_dataless = f"""
// Module of br_dataless
module {name} (
  input  clk,
  input  rst,
  // Input Channel
  input  ins_valid,
  output ins_ready,
  // Output Channel
  output outs_valid,
  input  outs_ready
);

  assign outs_valid = ins_valid;
  assign ins_ready = outs_ready;

endmodule
"""

    return body_br_dataless
