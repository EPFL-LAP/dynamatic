from generators.support.mc_support import generate_mc_support

def generate_mem_controller_storeless(name, params):
    # Number of input ports
    num_controls = params["num_controls"]
    num_loads = params["num_loads"]
    data_type = params["data_type"]
    addr_type = params["addr_type"]

    mc_support = generate_mc_support("TODO", {})

    mem_controller_storeless_body = f"""

// Module of mem_controller_storeless
module {name}(
  input                                     clk,
  input                                     rst,
  // start input control
  input                                     memStart_valid,
  output                                    memStart_ready,
  // end output control
  output                                    memEnd_valid,
  input                                     memEnd_ready,
  // "no more requests" input control
  input                                     ctrlEnd_valid,
  output                                    ctrlEnd_ready,
  // Load address input channels
  input  [({num_loads} * {addr_type}) - 1 : 0] ldAddr,
  input  [               {num_loads} - 1 : 0] ldAddr_valid,
  output [               {num_loads} - 1 : 0] ldAddr_ready,
  // Load data output channels
  output [({num_loads} * {data_type}) - 1 : 0] ldData,
  output [               {num_loads} - 1 : 0] ldData_valid,
  input  [               {num_loads} - 1 : 0] ldData_ready,
  // Interface to dual-port BRAM
  input  [              {data_type} - 1 : 0] loadData,
  output                                    loadEn,
  output [              {addr_type} - 1 : 0] loadAddr,
  output                                    storeEn,
  output [              {addr_type} - 1 : 0] storeAddr,
  output [              {data_type} - 1 : 0] storeData
);
  wire allRequestsDone;

  // No stores will ever be issused
  assign storeAddr = {{{addr_type}{{1'b0}}}};
  assign storeData = {{{data_type}{{1'b0}}}};
  assign storeEn   = 1'b0;

  // MC is "always done with stores"

  read_memory_arbiter #(
    .ARBITER_SIZE({num_loads}),
    .ADDR_TYPE  ({addr_type}),
    .DATA_TYPE  ({data_type})
  ) read_arbiter (
    .rst             (rst),
    .clk             (clk),
    .pValid          (ldAddr_valid),
    .ready           (ldAddr_ready),
    .address_in      (ldAddr),
    .nReady          (ldData_ready),
    .valid           (ldData_valid),
    .data_out        (ldData),
    .read_enable     (loadEn),
    .read_address    (loadAddr),
    .data_from_memory(loadData)
  );

  // NOTE: (lucas-rami) In addition to making sure there are no stores pending,
  // we should also check that there are no loads pending as well. To achieve 
  // this the control signals could simply start indicating the total number
  // of accesses in the block instead of just the number of stores.
  assign allRequestsDone = 1'b1;

  mc_control control (
    .rst            (rst),
    .clk            (clk),
    .memStart_valid (memStart_valid),
    .memStart_ready (memStart_ready),
    .memEnd_valid   (memEnd_valid),
    .memEnd_ready   (memEnd_ready),
    .ctrlEnd_valid  (ctrlEnd_valid),
    .ctrlEnd_ready  (ctrlEnd_ready),
    .allRequestsDone(allRequestsDone)
  );

endmodule
"""
    return mc_support + mem_controller_storeless_body
