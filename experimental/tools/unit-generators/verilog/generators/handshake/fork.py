from generators.handshake.dataless.datalessFork import generate_datalessFork

def generate_fork(name, params):
    size = params["size"]
    bitwidth = params["bitwidth"]

    if(bitwidth == 0):
        return generate_datalessFork(name, params)
    
    datalessFork_name = name + "_datalessFork"

    header = "`timescale 1ns/1ps\n"



    datalessFork = generate_datalessFork(datalessFork_name, params)
    Fork = f"""
// Module of Fork
module {name} #(
	parameter SIZE = {size},
	parameter DATA_TYPE = {bitwidth}
)(
	input  clk,
	input  rst,
  // Input Channel
	input  [DATA_TYPE - 1 : 0] ins,
	input  ins_valid,
  output ins_ready,
  // Output Channel
  output [SIZE * (DATA_TYPE) - 1 : 0] outs,
  output [SIZE - 1 : 0] outs_valid,
	input  [SIZE - 1 : 0] outs_ready
);

  {datalessFork_name} #(
    .SIZE(SIZE)
  ) control (
    .clk        (clk        ),
    .rst        (rst        ),
    .ins_valid  (ins_valid  ),
    .ins_ready  (ins_ready  ),
    .outs_valid (outs_valid ),
    .outs_ready (outs_ready )
  );

  // Broadcast the input data to all output channels
  assign outs = {{SIZE{{ins}}}};

endmodule
"""


    return header + datalessFork + Fork