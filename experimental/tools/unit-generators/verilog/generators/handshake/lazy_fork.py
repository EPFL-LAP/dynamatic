from generators.handshake.dataless.dataless_lazy_fork import generate_dataless_lazy_fork

def generate_lazy_fork(name, params):
    size = params["size"]
    bitwidth = params["bitwidth"]

    if(bitwidth == 0):
        return generate_dataless_lazy_fork(name, params)
    
    dataless_lazy_fork_name = name + "_dataless_lazy_fork"

    verilog_header = "`timescale 1ns/1ps\n"



    verilog_datalessFork = generate_dataless_lazy_fork(dataless_lazy_fork_name, params)
    verilog_lazy_fork = f"""
// Module of lazy_fork
`timescale 1ns/1ps
module {name} #(
  parameter SIZE = {size},
  parameter DATA_TYPE = {bitwidth}
)(
  input  clk,
	input  rst,
  // Input Channels
  input  [DATA_TYPE - 1 : 0] ins,
  input  ins_valid,
  output ins_ready,
  // Output Channel
  output [SIZE * (DATA_TYPE) - 1 : 0] outs,
	output [SIZE - 1 : 0] outs_valid,
	input  [SIZE - 1 : 0] outs_ready
);

  {dataless_lazy_fork_name} #(
    .SIZE(SIZE)
  ) control (
    .clk 			    (clk				        ),
    .rst 			    (rst				        ),
    .ins_valid 		(ins_valid			    ),
    .ins_ready    (ins_ready		      ),
    .outs_valid 	(outs_valid     	  ),
    .outs_ready 	(outs_ready         )
  );

  assign outs = {{SIZE{{ins}}}};

endmodule
"""


    return verilog_header + verilog_datalessFork + verilog_lazy_fork