from generators.handshake.dataless.dataless_lazy_fork import generate_dataless_lazy_fork

def generate_lazy_fork(name, params):
    size = params["size"]
    bitwidth = params["bitwidth"]

    if(bitwidth == 0):
        return generate_dataless_lazy_fork(name, {"size": size})

    dataless_lazy_fork_name = name + "_dataless_lazy_fork"

    header = "`timescale 1ns/1ps\n"



    datalessFork = generate_dataless_lazy_fork(dataless_lazy_fork_name, {"size": size})
    lazy_fork = f"""
// Module of lazy_fork
`timescale 1ns/1ps
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

  {dataless_lazy_fork_name} control (
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


    return header + datalessFork + lazy_fork