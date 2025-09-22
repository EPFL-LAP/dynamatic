from generators.handshake.dataless.datalessFork import generate_datalessFork

def generate_fork(name, params):
    size = params["size"]
    bitwidth = params["bitwidth"]

    if(bitwidth == 0):
        return generate_datalessFork(name, {"size": size})
    
    datalessFork_name = name + "_datalessFork"

    header = "`timescale 1ns/1ps\n"



    datalessFork = generate_datalessFork(datalessFork_name, {"size": size})
    Fork = f"""
// Module of Fork
module {name}(
	input  clk,
	input  rst,
  // Input Channel
	input  [{bitwidth} - 1 : 0] ins,
	input  ins_valid,
  output ins_ready,
  // Output Channel
  output [{size} * ({bitwidth}) - 1 : 0] outs,
  output [{size} - 1 : 0] outs_valid,
	input  [{size} - 1 : 0] outs_ready
);

  {datalessFork_name} control (
    .clk        (clk        ),
    .rst        (rst        ),
    .ins_valid  (ins_valid  ),
    .ins_ready  (ins_ready  ),
    .outs_valid (outs_valid ),
    .outs_ready (outs_ready )
  );

  // Broadcast the input data to all output channels
  assign outs = {{{size}{{ins}}}};

endmodule
"""


    return header + datalessFork + Fork