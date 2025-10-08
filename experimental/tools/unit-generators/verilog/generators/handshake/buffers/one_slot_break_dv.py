from generators.support.signal_manager import generate_concat_signal_manager
from generators.support.signal_manager.utils.concat import get_concat_extra_signals_bitwidth


def generate_one_slot_break_dv(name, params):
    bitwidth = params["bitwidth"]
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_one_slot_break_dv_signal_manager(name, bitwidth, extra_signals)
    if bitwidth == 0:
        return _generate_one_slot_break_dv_dataless(name)
    else:
        return _generate_one_slot_break_dv(name, bitwidth)


def _generate_one_slot_break_dv(name, bitwidth):

    one_slot_break_dv_dataless_name = name + "_dataless"
    one_slot_break_dv_dataless = _generate_one_slot_break_dv_dataless(
        one_slot_break_dv_dataless_name)

    one_slot_break_dv_body = f"""
// Module of one_slot_break_dv
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
  wire regEn, inputReady;
  reg [{bitwidth} - 1 : 0] dataReg = 0;
  
  // Instance of one_slot_break_dv_dataless to manage handshaking
  {one_slot_break_dv_dataless_name} control (
    .clk        (clk       ),
    .rst        (rst       ),
    .ins_valid  (ins_valid ),
    .ins_ready  (inputReady),
    .outs_valid (outs_valid),
    .outs_ready (outs_ready)
  );

  always @(posedge clk) begin
    if (rst) begin
      dataReg <= {{{bitwidth}{{1'b0}}}};
    end else if (regEn) begin
      dataReg <= ins;
    end
  end

  assign ins_ready = inputReady;
  assign regEn = inputReady & ins_valid;
  assign outs = dataReg;

endmodule

"""

    return one_slot_break_dv_dataless + one_slot_break_dv_body


def _generate_one_slot_break_dv_dataless(name):
    return f"""
// Module of one_slot_break_dv_dataless
module {name} (
  input  clk,
  input  rst,
  // Input channel
  input  ins_valid,
  output ins_ready,
  // Output channel
  output outs_valid,
  input  outs_ready
);
  // Define internal signals
  reg outputValid = 0;

  always @(posedge clk) begin
    if (rst) begin
      outputValid <= 0;
    end else begin
      outputValid <= ins_valid | (~outs_ready & outputValid);
    end
  end

  assign ins_ready = ~outputValid | outs_ready;
  assign outs_valid = outputValid;
  
endmodule
"""


def _generate_one_slot_break_dv_signal_manager(name, bitwidth, extra_signals):
    extra_signals_bitwidth = get_concat_extra_signals_bitwidth(extra_signals)
    return generate_concat_signal_manager(
        name,
        [{
            "name": "ins",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }],
        [{
            "name": "outs",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }],
        extra_signals,
        lambda name: _generate_one_slot_break_dv(name, bitwidth + extra_signals_bitwidth))
