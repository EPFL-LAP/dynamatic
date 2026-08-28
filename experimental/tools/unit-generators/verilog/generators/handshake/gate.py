from generators.handshake.join import generate_join


def generate_gate(name, params):
    size = params["size"]
    bitwidth = params["bitwidth"]
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        raise ValueError(
            "Verilog handshake.gate does not support extra signals yet"
        )

    return _generate_gate(name, size, bitwidth)


def _generate_gate(name, size, bitwidth):
    join_name = f"{name}_join"
    dependencies = generate_join(join_name, {"size": size})

    return dependencies + f"""
// Module of gate
module {name} (
  input  wire                    clk,
  input  wire                    rst,
  input  wire [{bitwidth} - 1:0] ins,
  input  wire [{size} - 1:0]     ins_valid,
  output wire [{size} - 1:0]     ins_ready,
  output wire [{bitwidth} - 1:0] outs,
  output wire                    outs_valid,
  input  wire                    outs_ready
);

  {join_name} control (
    .ins_valid  (ins_valid),
    .outs_ready (outs_ready),
    .ins_ready  (ins_ready),
    .outs_valid (outs_valid)
  );

  assign outs = ins;

endmodule
"""
