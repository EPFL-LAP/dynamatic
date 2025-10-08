from generators.support.signal_manager import generate_concat_signal_manager
from generators.support.signal_manager.utils.concat import get_concat_extra_signals_bitwidth


def generate_lazy_fork(name, params):
    # Number of output ports
    size = params["size"]

    bitwidth = params["bitwidth"]
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_lazy_fork_signal_manager(name, size, bitwidth, extra_signals)
    elif bitwidth == 0:
        return _generate_lazy_fork_dataless(name, size)
    else:
        return _generate_lazy_fork(name, size, bitwidth)

def _generate_lazy_fork(name, size, bitwidth):

    lazy_fork_dataless_name = name + "_lazy_fork_dataless"



    fork_dataless = _generate_lazy_fork_dataless(lazy_fork_dataless_name, size)
    lazy_fork = f"""
// Module of lazy_fork
module {name}(
  input  clk,
	input  rst,
  // Input Channels
  input  [{bitwidth} - 1 : 0] ins,
  input  ins_valid,
  output ins_ready,
  // Output Channel
  output [{size} * ({bitwidth}) - 1 : 0] outs,
	output [{size} - 1 : 0] outs_valid,
	input  [{size} - 1 : 0] outs_ready
);

  {lazy_fork_dataless_name} control (
    .clk 			    (clk				        ),
    .rst 			    (rst				        ),
    .ins_valid 		(ins_valid			    ),
    .ins_ready    (ins_ready		      ),
    .outs_valid 	(outs_valid     	  ),
    .outs_ready 	(outs_ready         )
  );

  assign outs = {{{size}{{ins}}}};

endmodule
"""


    return fork_dataless + lazy_fork

def _generate_lazy_fork_dataless(name, size):

    fork_dataless = f"""
// Module of lazy_fork_dataless
module {name}(
  input  clk,
	input  rst,
  // Input Channel
  input  ins_valid,
  output ins_ready,
  // Output Channels
  output reg [{size} - 1 : 0] outs_valid,
	input  [{size} - 1: 0] outs_ready
);
  wire allnReady;
  assign allnReady = &outs_ready;

  // Process to handle output valid signals based on input valid and output readiness
  integer i, j;
  reg [{size} - 1 : 0] tmp_ready;

  always @(*) begin
    tmp_ready = {{{size}{{1'b1}}}};

    for (i = 0; i < {size}; i = i + 1) begin
      for (j = 0; j < {size}; j = j + 1) begin
        if (i != j) begin
          tmp_ready[i] = tmp_ready[i] & outs_ready[j];
        end
      end
    end

    for (i = 0; i < {size}; i = i + 1) begin
      outs_valid[i] = ins_valid & tmp_ready[i];
    end
  end

  assign ins_ready = allnReady;

endmodule

"""


    return fork_dataless


def _generate_lazy_fork_signal_manager(name, size, bitwidth, extra_signals):
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
            "extra_signals": extra_signals,
            "size": size
        }],
        extra_signals,
        lambda name: _generate_lazy_fork(name, size, bitwidth + extra_signals_bitwidth))