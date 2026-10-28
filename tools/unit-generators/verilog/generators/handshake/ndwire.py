from generators.support.utils import data


def generate_ndwire(name, params):
    bitwidth = params["bitwidth"]
    extra_signals = params.get("extra_signals", None)

    return _generate_ndwire(name, bitwidth)


def _generate_ndwire(name, params):
    bitwidth = params["bitwidth"]

    ndwire = f"""
// Module of ndwire
module ndwire(
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

  ndwire_dataless #(
    .SIZE({bitwidth})
  ) control (
    .clk        (clk        ),
    .rst        (rst        ),
    .ins_valid  (ins_valid  ),
    .ins_ready  (ins_ready  ),
    .outs_valid (outs_valid ),
    .outs_ready (outs_ready )
  );
  
  assign outs = ins;

endmodule
"""

    return ndwire
