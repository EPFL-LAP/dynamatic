from generators.support.signal_manager import generate_default_signal_manager
from generators.support.utils import data


def generate_sink(name, params):
    bitwidth = params["bitwidth"]
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_sink_signal_manager(name, bitwidth, extra_signals)
    else:
        return _generate_sink(name, bitwidth)

def _generate_sink(name, bitwidth):

    empty_sink = f"""
// Module of sink
module {name} (
    input  clk,    
    input  rst,     
    input  ins_valid,  
    output ins_ready 
);
  assign ins_ready = 1'b1;

endmodule
"""
    non_empty_sink = f"""
// Module of sink
module {name}(
  input  clk,      
  input  rst,       
  input  [{bitwidth}-1:0] ins, 
  input  ins_valid, 
  output ins_ready 
);
  assign ins_ready = 1'b1;

endmodule

"""

    if bitwidth:
        return non_empty_sink
    else:
        return empty_sink


def _generate_sink_signal_manager(name, bitwidth, extra_signals):
    return generate_default_signal_manager(
        name,
        [{
            "name": "ins",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }],
        [],
        extra_signals,
        lambda name: _generate_sink(name, bitwidth))
