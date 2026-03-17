def generate_store(name, params):
    data_bitwidth = params["data_bitwidth"]
    addr_bitwidth = params["addr_bitwidth"]
    
    return _generate_store(name, data_bitwidth, addr_bitwidth)


def _generate_store(name, data_bitwidth, addr_bitwidth):
    return f"""
// Module of Store
module {name}(
  input  clk,
  input  rst,
  // Data from Circuit Channel
  input  [{data_bitwidth} - 1 : 0] dataIn,
  input  dataIn_valid,
  output dataIn_ready,
  // Address from Circuit Channel
  input  [{addr_bitwidth} - 1 : 0] addrIn,
  input  addrIn_valid,
  output addrIn_ready,
  // Data to Interface Channel
  output [{data_bitwidth} - 1 : 0] dataToMem,
  output dataToMem_valid,
  input  dataToMem_ready,
  // Address to Interface Channel
  output [{addr_bitwidth} - 1 : 0] addrOut,
  output addrOut_valid,
  input  addrOut_ready 
);

  // Data assignment
  assign dataToMem = dataIn;
  assign dataToMem_valid = dataIn_valid;
  assign dataIn_ready = dataToMem_ready;

  // Address assignment
  assign addrOut = addrIn;
  assign addrOut_valid = addrIn_valid;
  assign addrIn_ready = addrOut_ready;

endmodule
"""
