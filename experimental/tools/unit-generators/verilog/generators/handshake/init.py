def generate_init(name, params):
    bitwidth = params["bitwidth"]
    extra_signals = params.get("extra_signals", None)
    initial_value = params.get("initial_value", 0)

    if extra_signals:
        raise ValueError(
            "Verilog handshake.init does not support extra signals yet"
        )

    if bitwidth == 0:
        return _generate_init_dataless(name)

    return _generate_init(name, bitwidth, initial_value)


def _generate_init_dataless(name):
    return f"""
// Module of init_dataless
module {name} (
  input  wire clk,
  input  wire rst,
  input  wire ins_valid,
  output wire ins_ready,
  output wire outs_valid,
  input  wire outs_ready
);

  reg fullReg;
  wire outputValid;

  assign outputValid = ins_valid | fullReg;

  always @(posedge clk) begin
    if (rst)
      fullReg <= 1'b1;
    else
      fullReg <= outputValid & ~outs_ready;
  end

  assign ins_ready  = ~fullReg;
  assign outs_valid = outputValid;

endmodule
"""


def _generate_init(name, bitwidth, initial_value):
    dataless_name = f"{name}_dataless"
    dependencies = _generate_init_dataless(dataless_name)

    return dependencies + f"""
// Module of init
module {name} (
  input  wire                    clk,
  input  wire                    rst,
  input  wire [{bitwidth} - 1:0] ins,
  input  wire                    ins_valid,
  output wire                    ins_ready,
  output wire [{bitwidth} - 1:0] outs,
  output wire                    outs_valid,
  input  wire                    outs_ready
);

  wire regNotFull;
  wire regEnable;
  reg [{bitwidth} - 1:0] dataReg;

  assign regEnable = regNotFull & ins_valid & ~outs_ready;

  {dataless_name} control (
    .clk        (clk),
    .rst        (rst),
    .ins_valid  (ins_valid),
    .ins_ready  (regNotFull),
    .outs_valid (outs_valid),
    .outs_ready (outs_ready)
  );

  always @(posedge clk) begin
    if (rst)
      dataReg <= {initial_value};
    else if (regEnable)
      dataReg <= ins;
  end

  assign outs      = regNotFull ? ins : dataReg;
  assign ins_ready = regNotFull;

endmodule
"""
