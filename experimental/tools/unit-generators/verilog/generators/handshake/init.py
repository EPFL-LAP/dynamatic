def generate_init(name, params):
    bitwidth = params["bitwidth"]
    # the value the output holds before the first input arrives
    initial_value = params.get("initial_value", 0)

    if bitwidth == 0:
        return _generate_init_dataless(name)
    else:
        return _generate_init(name, bitwidth, initial_value)


def _generate_init(name, bitwidth, initial_value):

    init_dataless_name = name + "_init_dataless"
    init_dataless = _generate_init_dataless(init_dataless_name)

    init_body = f"""
// Module of init
module {name}(
  input  clk,
  input  rst,
  // Input Channel
  input  [{bitwidth} - 1 : 0] ins,
  input  ins_valid,
  output ins_ready,
  // Output Channel
  output [{bitwidth} - 1 : 0] outs,
  output outs_valid,
  input  outs_ready
);
  // Signal Definition
  wire regEnable, regNotFull;
  reg [{bitwidth} - 1 : 0] dataReg = {initial_value};

  // Instantiate control logic part
  {init_dataless_name} control (
    .clk        (clk       ),
    .rst        (rst       ),
    .ins_valid  (ins_valid ),
    .ins_ready  (regNotFull),
    .outs_valid (outs_valid),
    .outs_ready (outs_ready)
  );

  assign regEnable = regNotFull & ins_valid & ~outs_ready;

  always @(posedge clk) begin
    if (rst) begin
      dataReg <= {initial_value};
    end else if (regEnable) begin
      dataReg <= ins;
    end
  end

  // Output Assignment
  assign outs = regNotFull ? ins : dataReg;

  assign ins_ready = regNotFull;

endmodule

"""
    return init_dataless + init_body


def _generate_init_dataless(name):
    return f"""
// Module of init_dataless
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
  // the register starts full, so one token comes out before any input
  reg fullReg = 1;

  always @(posedge clk) begin
    if (rst) begin
      fullReg <= 1;
    end else begin
      fullReg <= (ins_valid | fullReg) & ~outs_ready;
    end
  end

  assign ins_ready = ~fullReg;
  assign outs_valid = ins_valid | fullReg;

endmodule
"""
