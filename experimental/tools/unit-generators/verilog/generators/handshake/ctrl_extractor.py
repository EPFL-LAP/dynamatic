def generate_ctrl_extractor(name, params):
    bitwidth = params["bitwidth"]
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        raise ValueError(
            "Verilog handshake.ctrl_extractor does not support extra signals yet"
        )

    return _generate_ctrl_extractor(name, bitwidth)


def _generate_ctrl_extractor(name, bitwidth):
    return f"""
// Module of ctrl_extractor
module {name} (
  input  wire                    clk,
  input  wire                    rst,
  input  wire [{bitwidth} - 1:0] ins,
  input  wire                    ins_valid,
  output wire                    ins_ready,
  output wire                    outs_valid,
  input  wire                    outs_ready
);

  assign outs_valid = ins_valid;
  assign ins_ready  = outs_ready;

endmodule
"""
