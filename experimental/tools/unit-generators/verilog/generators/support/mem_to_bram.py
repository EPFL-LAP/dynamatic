def generate_mem_to_bram(name, params):
    # Number of input ports
    data_width = params["data_width"]
    addr_width = params["addr_width"]

    return f"""
`timescale 1ns / 1ps

// Module of mem_to_bram
module {name} #(
  parameter DATA_WIDTH = {data_width},
  parameter ADDR_WIDTH = {addr_width}
) (
  // Inputs from circuit
  input                       loadEn,
  input  [ADDR_WIDTH - 1 : 0] loadAddr,
  input                       storeEn,
  input  [ADDR_WIDTH - 1 : 0] storeAddr,
  input  [DATA_WIDTH - 1 : 0] storeData,
  // Inputs from BRAM
  input  [DATA_WIDTH - 1 : 0] din0,
  input  [DATA_WIDTH - 1 : 0] din1,
  // Outputs to BRAM
  output                      ce0,
  output                      we0,
  output [ADDR_WIDTH - 1 : 0] address0,
  output [DATA_WIDTH - 1 : 0] dout0,
  output                      ce1,
  output                      we1,
  output [ADDR_WIDTH - 1 : 0] address1,
  output [DATA_WIDTH - 1 : 0] dout1,
  // Outputs back to circuit
  output [DATA_WIDTH - 1 : 0] loadData
);
  // Store request
  assign ce0 = storeEn;
  assign we0 = storeEn;
  assign address0 = storeAddr;
  assign dout0 = storeData;

  // Load request
  assign ce1 = loadEn;
  assign we1 = 1'b0;  // Write enable is always 0 for load operations
  assign address1 = loadAddr;
  assign dout1 = {{DATA_WIDTH{{1'b0}}}};  // Data output is zero since no data is written during loads

  // Data back to circuit from BRAM
  assign loadData = din1;

endmodule
"""
