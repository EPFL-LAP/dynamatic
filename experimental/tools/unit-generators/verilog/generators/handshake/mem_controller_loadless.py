from generators.support.mc_support import generate_mc_support

def generate_mem_controller_loadless(name, params):
    # Number of input ports
    num_controls = params["num_controls"]
    num_stores = params["num_stores"]
    data_type = params["data_type"]
    addr_type = params["addr_type"]

    
    header = "`timescale 1ns / 1ps\n"

    mc_support = generate_mc_support("TODO", {})

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
  write_memory_arbiter #(
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
    return header + mc_support + mem_controller_loadless_body
