from generators.support.signal_manager import generate_default_signal_manager

def generate_constant(name, params):
    bitwidth = params["bitwidth"]
    value = params["value"]
    extra_signals = params.get("extra_signals", None)
    print(f"Generating constant {name} with value {value} and bitwidth {bitwidth}")

    if extra_signals:
        return _generate_constant_signal_manager(name, value, bitwidth, extra_signals)
    else:
        return _generate_constant(name, value, bitwidth)

def _generate_constant(name, value, bitwidth):

    constant = f"""
// Module of constant

module {name}(
  input                       clk,
  input                       rst,
  // Input Channel
  input                       ctrl_valid,
  output                      ctrl_ready,
  // Output Channel
  output [{bitwidth} - 1 : 0] outs,
  output                      outs_valid,
  input                       outs_ready
);
  assign outs       = {bitwidth}'b{value};
  assign outs_valid = ctrl_valid;
  assign ctrl_ready = outs_ready;

endmodule
"""
    return constant

def _generate_constant_signal_manager(name, value, bitwidth, extra_signals):
    return generate_default_signal_manager(
        name,
        [{
            "name": "ctrl",
            "bitwidth": 0,
            "extra_signals": extra_signals
        }],
        [{
            "name": "outs",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }],
        extra_signals,
        lambda name: _generate_constant(name, value, bitwidth))
