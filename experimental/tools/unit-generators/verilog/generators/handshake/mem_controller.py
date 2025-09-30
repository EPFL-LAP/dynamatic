from generators.support.mc_support import generate_mc_support
from generators.handshake.mem_controller_loadless import generate_mem_controller_loadless

def generate_mem_controller(name, params):
    # Number of input ports
    num_controls = params["num_controls"]
    num_loads = params["num_loads"]
    num_stores = params["num_stores"]
    data_type = params["data_type"]
    addr_type = params["addr_type"]

    
    header = "`timescale 1ns / 1ps\n"

    mc_support = generate_mc_support("TODO", {})
  
    mem_controller_loadless_name = name + "_mem_controller_loadless"
    mem_controller_loadless = generate_mem_controller_loadless(mem_controller_loadless_name, {"num_controls": num_controls, "num_stores": num_stores, "data_type": data_type, "addr_type": addr_type})

    mem_controller_body = f"""

// Module of mem_controller
module {name}(
  input                                      clk,
  input                                      rst,
  // start input control
  input                                      memStart_valid,
  output                                     memStart_ready,
  // end output control
  output                                     memEnd_valid,
  input                                      memEnd_ready,
  // "no more requests" input control
  input                                      ctrlEnd_valid,
  output                                     ctrlEnd_ready,
  // Control Input Channels
  input  [      ({num_controls} * 32) - 1 : 0] ctrl,
  input  [             {num_controls} - 1 : 0] ctrl_valid,
  output [             {num_controls} - 1 : 0] ctrl_ready,
  // Load address input channels
  input  [ ({num_loads} * {addr_type}) - 1 : 0] ldAddr,
  input  [                {num_loads} - 1 : 0] ldAddr_valid,
  output [                {num_loads} - 1 : 0] ldAddr_ready,
  // Load data output channels
  output [ ({num_loads} * {data_type}) - 1 : 0] ldData,
  output [                {num_loads} - 1 : 0] ldData_valid,
  input  [                {num_loads} - 1 : 0] ldData_ready,
  // Store Address Input Channels
  input  [({num_stores} * {addr_type}) - 1 : 0] stAddr,
  input  [               {num_stores} - 1 : 0] stAddr_valid,
  output [               {num_stores} - 1 : 0] stAddr_ready,
  // Store Data Input Channels
  input  [({num_stores} * {data_type}) - 1 : 0] stData,
  input  [               {num_stores} - 1 : 0] stData_valid,
  output [               {num_stores} - 1 : 0] stData_ready,
  // Interface to Dual-port BRAM
  input  [               {data_type} - 1 : 0] loadData,
  output                                     loadEn,
  output [               {addr_type} - 1 : 0] loadAddr,
  output                                     storeEn,
  output [               {addr_type} - 1 : 0] storeAddr,
  output [               {data_type} - 1 : 0] storeData
);
  // Internal signal declarations
  wire                      dropLoadEn;
  wire [{addr_type} - 1 : 0] dropLoadAddr;
  wire [{data_type} - 1 : 0] dropLoadData;

  {mem_controller_loadless_name} stores (
    .clk           (clk),
    .rst           (rst),
    .memStart_valid(memStart_valid),
    .memStart_ready(memStart_ready),
    .memEnd_valid  (memEnd_valid),
    .memEnd_ready  (memEnd_ready),
    .ctrlEnd_valid (ctrlEnd_valid),
    .ctrlEnd_ready (ctrlEnd_ready),
    .ctrl          (ctrl),
    .ctrl_valid    (ctrl_valid),
    .ctrl_ready    (ctrl_ready),
    .stAddr        (stAddr),
    .stAddr_valid  (stAddr_valid),
    .stAddr_ready  (stAddr_ready),
    .stData        (stData),
    .stData_valid  (stData_valid),
    .stData_ready  (stData_ready),
    .loadData      (dropLoadData),
    .loadEn        (dropLoadEn),
    .loadAddr      (dropLoadAddr),
    .storeEn       (storeEn),
    .storeAddr     (storeAddr),
    .storeData     (storeData)
  );

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

endmodule
"""
    return header + mem_controller_loadless + mc_support + mem_controller_body
