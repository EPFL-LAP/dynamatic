from generators.support.mc_support import generate_mc_support


def generate_mem_controller(name, params):
    # Number of input ports
    num_controls = params["num_controls"]
    num_loads = params["num_loads"]
    num_stores = params["num_stores"]
    data_type = params["data_bitwidth"]
    addr_type = params["addr_bitwidth"]

    if num_controls == 0 and num_loads > 0 and num_stores == 0:
        return _generate_mem_controller_storeless(name, num_loads, addr_type, data_type)
    elif num_controls > 0 and num_loads == 0 and num_stores > 0:
        return _generate_mem_controller_loadless(name, num_controls, num_stores, addr_type, data_type)
    elif num_controls > 0 and num_loads > 0 and num_stores > 0:
        return _generate_mem_controller_mixed(name, num_controls, num_loads, num_stores, addr_type, data_type)


def _generate_mem_controller_mixed(name, num_controls, num_loads, num_stores, addr_type, data_type):
    mc_support_name = name + "_mc_support"
    mc_support = generate_mc_support(mc_support_name, {})

    mem_controller_loadless_name = name + "_mem_controller_loadless"
    mem_controller_loadless = _generate_mem_controller_loadless(
        mem_controller_loadless_name, num_controls, num_stores, addr_type, data_type)

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

  {mc_support_name}_read_memory_arbiter #(
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
    return mem_controller_loadless + mc_support + mem_controller_body


def _generate_mem_controller_storeless(name, num_loads, addr_type, data_type):

    mc_support_name = name + "_mc_support"
    mc_support = generate_mc_support(mc_support_name, {})

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

  {mc_support_name}_read_memory_arbiter #(
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


def _generate_mem_controller_loadless(name, num_controls, num_stores, addr_type, data_type):
    mc_support_name = name + "_mc_support"
    mc_support = generate_mc_support(mc_support_name, {})

    mem_controller_loadless_body = f"""

// Module of mem_controller_loadless
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
  // Terminology:
  // Access ports    : circuit to memory_controller;
  // Interface ports : memory_controller to memory_interface (e.g., BRAM/AXI);

  // TODO: The size of this counter should be configurable
  wire [31 : 0] remainingStores;
  // Indicating the store interface port that there is a valid store request
  // (currently not used).
  wire [{num_stores} - 1 : 0] interface_port_valid;
  // Indicating a store port has both a valid data and a valid address.
  wire [{num_stores} - 1 : 0] store_access_port_complete_request;
  // Indicating the store port is selected by the arbiter.
  wire [{num_stores} - 1 : 0] store_access_port_selected;
  wire allRequestsDone;

  // Local Parameter
  localparam [31:0] zeroStore = 32'b0;
  localparam [{num_controls}-1:0] zeroCtrl = {{{num_controls}{{1'b0}}}};

  assign loadEn   = 0;
  assign loadAddr = {{{addr_type}{{1'b0}}}};

  // A store request is complete if both address and data are valid.
  assign store_access_port_complete_request = stAddr_valid & stData_valid;

  // Instantiate write memory arbiter
  {mc_support_name}_write_memory_arbiter #(
    .ARBITER_SIZE({num_stores}),
    .ADDR_TYPE  ({addr_type}),
    .DATA_TYPE  ({data_type})
  ) write_arbiter (
    .rst           (rst),
    .clk           (clk),
    .pValid        (store_access_port_complete_request),
    .ready         (store_access_port_selected),
    .address_in    (stAddr),
    .data_in       (stData),
    .nReady        ({{{num_stores}{{1'b1}}}}),
    .valid         (interface_port_valid),
    .write_enable  (storeEn),
    .write_address (storeAddr),
    .data_to_memory(storeData)
  );

  assign stData_ready = store_access_port_selected;
  assign stAddr_ready = store_access_port_selected;
  assign ctrl_ready   = {{{num_controls}{{1'b1}}}};

  integer          i;
  reg     [31 : 0] counter = 32'd0;

  // Counting Stores
  always @(posedge clk) begin
    if (rst) begin
      counter = 32'd0;
    end else begin
      for (i = 0; i <= {num_controls} - 1; i = i + 1) begin
        if (ctrl_valid[i]) begin
          counter = counter + ctrl[i*32+:32];
        end
      end
      if (storeEn) begin
        counter = counter - 1;
      end
    end
  end

  assign remainingStores = counter;

  // NOTE: (lucas-rami) In addition to making sure there are no stores pending,
  // we should also check that there are no loads pending as well. To achieve 
  // this the control signals could simply start indicating the total number
  // of accesses in the block instead of just the number of stores.
  assign allRequestsDone = (remainingStores == zeroStore && ctrl_valid == zeroCtrl) ? 1'b1 : 1'b0;

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
    return mc_support + mem_controller_loadless_body
