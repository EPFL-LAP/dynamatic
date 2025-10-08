from generators.support.signal_manager import generate_concat_signal_manager
from generators.support.signal_manager.utils.concat import get_concat_extra_signals_bitwidth


def generate_shift_reg_break_dv(name, params):
    bitwidth = params["bitwidth"]
    num_slots = params["num_slots"]

    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_shift_reg_break_dv_signal_manager(name, num_slots, bitwidth, extra_signals)
    if bitwidth == 0:
        return _generate_shift_reg_break_dv_dataless(name, num_slots)
    else:
        return _generate_shift_reg_break_dv(name, num_slots, bitwidth)


def _generate_shift_reg_break_dv(name, params):

    num_slots = params["num_slots"]
    bitwidth = params["bitwidth"]

    shift_reg_break_dv_dataless_name = "shift_reg_break_dv_dataless"
    shift_reg_break_dv_dataless = _generate_shift_reg_break_dv_dataless(
        shift_reg_break_dv_dataless_name, num_slots)

    shift_reg_break_dv_body = f"""
// Module of shift_reg_break_dv

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

  // Internal signals
  wire regEn, inputReady;
  reg [{bitwidth} - 1 : 0] Memory [0 : {num_slots} - 1];
  
  // Instance of shift_reg_break_dv_dataless to manage handshaking
  {shift_reg_break_dv_dataless_name} control (
    .clk        (clk        ),
    .rst        (rst        ),
    .ins_valid  (ins_valid  ),
    .ins_ready  (inputReady ),
    .outs_valid (outs_valid ),
    .outs_ready (outs_ready )
  );
  
  // See 'docs/Specs/Buffering/Buffering.md'
  // All the slots share a single handshake control and thus 
  // accept or stall inputs together.
  integer i;
  always @(posedge clk) begin
    if (regEn) begin
      for (i = {num_slots} - 1; i > 0; i = i - 1) begin
        Memory[i] <= Memory[i - 1];
      end
      Memory[0] <= ins;
    end
  end
  
  assign regEn     = inputReady;
  assign ins_ready = inputReady;
  assign outs      = Memory[{num_slots} - 1];

endmodule
"""

    return shift_reg_break_dv_dataless + shift_reg_break_dv_body


def _generate_shift_reg_break_dv_dataless(name, num_slots):

    return f"""
// Module of shift_reg_break_dv_dataless

module {name}(
  input  clk,
  input  rst,
  // Inputs
  input  ins_valid,
  input  outs_ready,
  // Outputs
  output outs_valid,
  output ins_ready
);

  // Internal signals
  reg  [{num_slots}-1:0] valid_reg;
  wire             regEn;

  // See 'docs/Specs/Buffering/Buffering.md'
  // All the slots share a single handshake control and thus 
  // accept or stall inputs together.
  always @(posedge clk) begin
    if (rst) begin
      valid_reg <= {{{num_slots}{{1'b0}}}};
    end else begin
      if (regEn) begin
        valid_reg[{num_slots}-1:1] <= valid_reg[{num_slots}-2:0];
        valid_reg[0]         <= ins_valid;
      end
    end
  end

  assign outs_valid = valid_reg[{num_slots}-1];
  assign regEn      = ~outs_valid | outs_ready;
  assign ins_ready  = regEn;

endmodule
"""


def _generate_shift_reg_break_dv_signal_manager(name, num_slots, bitwidth, extra_signals):
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
        lambda name: _generate_shift_reg_break_dv(name, num_slots, bitwidth + extra_signals_bitwidth))
