`timescale 1ns / 1ps
module mem_controller_li_storeless_burst #(
  parameter NUM_BURST_LOADS  = 1,
  parameter DATA_TYPE = 32,
  parameter ADDR_TYPE = 32,
  parameter BURST_TYPE = 32
) (
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
  input  [(NUM_BURST_LOADS * ADDR_TYPE) - 1 : 0] ldAddr,
  input  [               NUM_BURST_LOADS - 1 : 0] ldAddr_valid,
  output [               NUM_BURST_LOADS - 1 : 0] ldAddr_ready,
  // Load burst length input channels
  input  [(NUM_BURST_LOADS * BURST_TYPE) - 1 : 0] loadBurstLength,
  input  [               NUM_BURST_LOADS - 1 : 0] loadBurstLength_valid,
  output [               NUM_BURST_LOADS - 1 : 0] loadBurstLength_ready,
  // Load data output channels
  output [(NUM_BURST_LOADS * DATA_TYPE) - 1 : 0] ldData,
  output [               NUM_BURST_LOADS - 1 : 0] ldData_valid,
  input  [               NUM_BURST_LOADS - 1 : 0] ldData_ready,
  // Interface to Dual-port BRAM
  // Load Data from Interface
  input  [               DATA_TYPE - 1 : 0] loadData,
  input                                     loadData_valid,
  output                                     loadData_ready,
  // Load Address to Interface
  output [               ADDR_TYPE - 1 : 0] loadAddr,
  output                                     loadAddr_valid,
  input                                      loadAddr_ready,
  // Load Burst length to Interface
  output [               BURST_TYPE - 1 : 0] loadBurstLen,
  // Store Address to Interface
  output [               ADDR_TYPE - 1 : 0] storeAddr,
  output                                     storeAddr_valid,
  input                                      storeAddr_ready,
  // Store Data to Interface
  output [               DATA_TYPE - 1 : 0] storeData,
  output                                     storeData_valid,
  input                                      storeData_ready,
  // Store Burst length to Interface
  output [               BURST_TYPE - 1 : 0] storeBurstLen
);
  wire allRequestsDone;

  // No stores will ever be issused
  assign storeAddr = {ADDR_TYPE{1'b0}};
  assign storeData = {DATA_TYPE{1'b0}};
  assign storeAddr_valid = 1'b0;
  assign storeData_valid = 1'b0;
  assign storeBurstLen = {BURST_TYPE{1'b0}};

  // MC is "always done with stores"

  read_li_memory_arbiter_burst #(
    .ARBITER_SIZE(NUM_BURST_LOADS),
    .ADDR_TYPE  (ADDR_TYPE),
    .DATA_TYPE  (DATA_TYPE),
    .BURST_TYPE (BURST_TYPE)
  ) read_arbiter (
    .rst             (rst),
    .clk             (clk),
    .pValid          (ldAddr_valid),
    .ready           (ldAddr_ready),
    .address_in      (ldAddr),
    .pValid_burst (loadBurstLength_valid),
    .ready_burst (loadBurstLength_ready),
    .burst_length_in (loadBurstLength),
    .nReady          (ldData_ready),
    .valid           (ldData_valid),
    .data_out        (ldData),
    .ready_data          (loadData_ready),
    .pValid_data           (loadData_valid),
    .data_from_memory(loadData),
    .read_address    (loadAddr),
    .nReady_address          (loadAddr_ready),
    .valid_address           (loadAddr_valid),
    .read_burst_length (loadBurstLen)
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
