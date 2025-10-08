from generators.support.signal_manager import generate_default_signal_manager


def generate_source(name, params):
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_source_signal_manager(name, extra_signals)
    else:
        return _generate_source(name)


def _generate_source(name):
    return f"""
// Module of source
module {name} (
  input  clk,        
  input  rst,        
  input  outs_ready, 
  output outs_valid 
);
  assign outs_valid = 1;

endmodule

"""


def _generate_source_signal_manager(name, extra_signals):
    return generate_default_signal_manager(
        name,
        [],
        [{
            "name": "outs",
            "bitwidth": 0,
            "extra_signals": extra_signals
        }],
        extra_signals,
        lambda name: _generate_source(name))
