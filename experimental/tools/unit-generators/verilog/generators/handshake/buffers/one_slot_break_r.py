from generators.support.signal_manager import generate_concat_signal_manager
from generators.support.signal_manager.utils.concat import get_concat_extra_signals_bitwidth


def generate_one_slot_break_r(name, params):
    bitwidth = params["bitwidth"]
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_one_slot_break_r_signal_manager(name, bitwidth, extra_signals)
    elif bitwidth == 0:
        return _generate_one_slot_break_r_dataless(name)
    else:
        return _generate_one_slot_break_r(name, bitwidth)


def _generate_one_slot_break_r(name, bitwidth):

    one_slot_break_r_dataless_name = name + "_one_slot_break_r_dataless"
    one_slot_break_r_dataless = _generate_one_slot_break_r_dataless(
        one_slot_break_r_dataless_name)

    one_slot_break_r_body = f"""
// Module of one_slot_break_r
module {name}(
	input  clk,
	input  rst,
  // Input Channel
	input  [{bitwidth} - 1 : 0] ins,
	input  ins_valid,
  output ins_ready,
  // Output Channel
  output [{bitwidth} - 1 : 0]	outs,
  output outs_valid,
	input  outs_ready
);
	// Signal Definition
	wire regEnable, regNotFull;
	reg [{bitwidth} - 1 : 0] dataReg = 0;

	// Instantiate control logic part
	{one_slot_break_r_dataless_name} control (
		.clk		    (clk	     ),
		.rst		    (rst	     ),
		.ins_valid	(ins_valid ),
    .ins_ready	(regNotFull),
		.outs_valid	(outs_valid),
		.outs_ready	(outs_ready)
	);

	assign regEnable = regNotFull & ins_valid & ~outs_ready;

	always @(posedge clk) begin
		if (rst) begin
			dataReg <= 0;
		end else if (regEnable) begin
			dataReg <= ins;
		end
	end

	// Output Assignment
	assign outs = regNotFull ? ins : dataReg;

	assign ins_ready = regNotFull;

endmodule

"""
    return one_slot_break_r_dataless + one_slot_break_r_body


def _generate_one_slot_break_r_dataless(name):
    return f"""
// Module of one_slot_break_r
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
	reg fullReg = 0;
	
	always @(posedge clk) begin
		if (rst) begin
			fullReg <= 0;
		end else begin
			fullReg <= (ins_valid | fullReg) & ~outs_ready;
		end
	end

	assign ins_ready = ~fullReg;
	assign outs_valid = ins_valid | fullReg;

endmodule
"""


def _generate_one_slot_break_r_signal_manager(name, bitwidth, extra_signals):
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
        lambda name: _generate_one_slot_break_r(name, bitwidth + extra_signals_bitwidth))
