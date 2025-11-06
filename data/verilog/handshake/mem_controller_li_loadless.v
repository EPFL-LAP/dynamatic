`timescale 1ns / 1ps
module mem_controller_li_loadless #(
  parameter NUM_CONTROLS = 1,
  parameter NUM_STORES   = 1,
  parameter DATA_TYPE   = 32,
  parameter ADDR_TYPE   = 32,
  parameter BURST_TYPE  = 32
) (
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
  input  [      (NUM_CONTROLS * 32) - 1 : 0] ctrl,
  input  [             NUM_CONTROLS - 1 : 0] ctrl_valid,
  output [             NUM_CONTROLS - 1 : 0] ctrl_ready,
  // Store Address Input Channels
  input  [(NUM_STORES * ADDR_TYPE) - 1 : 0] stAddr,
  input  [               NUM_STORES - 1 : 0] stAddr_valid,
  output [               NUM_STORES - 1 : 0] stAddr_ready,
  // Store Data Input Channels
  input  [(NUM_STORES * DATA_TYPE) - 1 : 0] stData,
  input  [               NUM_STORES - 1 : 0] stData_valid,
  output [               NUM_STORES - 1 : 0] stData_ready,
  // Interface to LI Dual-port BRAM
  output [               ADDR_TYPE - 1 : 0] loadAddr,
  output                                     loadAddr_valid,
  input                                      loadAddr_ready,
  output [               BURST_TYPE - 1 : 0] loadBurstLen,
  input  [               DATA_TYPE - 1 : 0] loadData,
  input                                     loadData_valid,
  output                                     loadData_ready,
  output [               ADDR_TYPE - 1 : 0] storeAddr,
  output                                     storeAddr_valid,
  input                                      storeAddr_ready,
  output [               BURST_TYPE - 1 : 0] storeBurstLen,
  output [               DATA_TYPE - 1 : 0] storeData,
  output                                     storeData_valid,
  input                                      storeData_ready
);
  // Terminology:
  // Access ports    : circuit to memory_controller;
  // Interface ports : memory_controller to memory_interface (e.g., BRAM/AXI);

  // TODO: The size of this counter should be configurable
  wire [31 : 0] remainingStores;
  // Indicating a store port has both a valid data and a valid address.
  wire [NUM_STORES - 1 : 0] store_access_port_complete_request;
  // Indicating the store port is selected by the arbiter.
  wire [NUM_STORES - 1 : 0] store_access_port_selected;
  wire allRequestsDone;

  // Local Parameter
  localparam [31:0] zeroStore = 32'b0;
  localparam [NUM_CONTROLS-1:0] zeroCtrl = {NUM_CONTROLS{1'b0}};

  assign loadAddr_valid   = 0;
  assign loadAddr = {ADDR_TYPE{1'b0}};

  assign loadBurstLen = {BURST_TYPE{1'b0}};
  assign storeBurstLen = 1;

  // A store request is complete if both address and data are valid.
  assign store_access_port_complete_request = stAddr_valid & stData_valid;

  // Instantiate latency-insensitive write memory arbiter
  write_li_memory_arbiter #(
    .ARBITER_SIZE(NUM_STORES),
    .ADDR_TYPE  (ADDR_TYPE),
    .DATA_TYPE  (DATA_TYPE)
  ) write_arbiter (
    .rst           (rst),
    .clk           (clk),
    .pValid        (store_access_port_complete_request),
    .ready         (store_access_port_selected),
    .address_in    (stAddr),
    .data_in       (stData),
    .nReady_address        (storeAddr_ready),
    .nReady_data           (storeData_ready),
    .valid_address         (storeAddr_valid),
    .valid_data            (storeData_valid),
    .write_address (storeAddr),
    .data_to_memory(storeData)
  );

  assign stData_ready = store_access_port_selected;
  assign stAddr_ready = store_access_port_selected;
  assign ctrl_ready   = {NUM_CONTROLS{1'b1}};

  integer          i;
  reg     [31 : 0] counter = 32'd0;

  // Counting Stores
  always @(posedge clk) begin
    if (rst) begin
      counter = 32'd0;
    end else begin
      for (i = 0; i <= NUM_CONTROLS - 1; i = i + 1) begin
        if (ctrl_valid[i]) begin
          counter = counter + ctrl[i*32+:32];
        end
      end
      if (storeData_valid & storeAddr_valid) begin
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
