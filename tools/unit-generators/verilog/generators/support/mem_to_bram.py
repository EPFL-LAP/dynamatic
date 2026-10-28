def generate_mem_to_bram(name, params):
    # Number of input ports
    data_width = params["data_width"]
    addr_width = params["addr_width"]

    return f"""
// Module of mem_to_bram
module {name}(
  // Inputs from circuit
  input                       loadEn,
  input  [{addr_width} - 1 : 0] loadAddr,
  input                       storeEn,
  input  [{addr_width} - 1 : 0] storeAddr,
  input  [{data_width} - 1 : 0] storeData,
  // Inputs from BRAM
  input  [{data_width} - 1 : 0] din0,
  input  [{data_width} - 1 : 0] din1,
  // Outputs to BRAM
  output                      ce0,
  output                      we0,
  output [{addr_width} - 1 : 0] address0,
  output [{data_width} - 1 : 0] dout0,
  output                      ce1,
  output                      we1,
  output [{addr_width} - 1 : 0] address1,
  output [{data_width} - 1 : 0] dout1,
  // Outputs back to circuit
  output [{data_width} - 1 : 0] loadData
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
  assign dout1 = {{{data_width}{{1'b0}}}};  // Data output is zero since no data is written during loads

  // Data back to circuit from BRAM
  assign loadData = din1;

endmodule
"""
