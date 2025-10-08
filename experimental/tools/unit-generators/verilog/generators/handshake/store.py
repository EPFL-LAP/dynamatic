from generators.support.signal_manager.utils.entity import generate_entity

def generate_store(name, params):
    data_bitwidth = params["data_bitwidth"]
    addr_bitwidth = params["addr_bitwidth"]
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_store_signal_manager(name, data_bitwidth, addr_bitwidth, extra_signals)
    else:
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


def _generate_store_signal_manager(name, data_bitwidth, addr_bitwidth, extra_signals):
    # Discard extra signals

    inner_name = f"{name}_inner"
    inner = _generate_store(inner_name, data_bitwidth, addr_bitwidth)

    entity = generate_entity(name, [{
        "name": "dataIn",
        "bitwidth": data_bitwidth,
        "extra_signals": extra_signals
    }, {
        "name": "addrIn",
        "bitwidth": addr_bitwidth,
        "extra_signals": extra_signals
    }], [{
        "name": "dataToMem",
        "bitwidth": data_bitwidth,
        "extra_signals": {}
    }, {
        "name": "addrOut",
        "bitwidth": addr_bitwidth,
        "extra_signals": {}
    }])

    architecture = f"""
  {inner_name} inner (
      .clk(clk),
      .rst(rst),
      .dataIn(dataIn),
      .dataIn_valid(dataIn_valid),
      .dataIn_ready(dataIn_ready),
      .addrIn(addrIn),
      .addrIn_valid(addrIn_valid),
      .addrIn_ready(addrIn_ready),
      .dataToMem(dataToMem),
      .dataToMem_valid(dataToMem_valid),
      .dataToMem_ready(dataToMem_ready),
      .addrOut(addrOut),
      .addrOut_valid(addrOut_valid),
      .addrOut_ready(addrOut_ready)
  );
  endmodule
"""

    return inner + entity + architecture
