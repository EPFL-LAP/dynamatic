`timescale 1ns / 1ps
module mem_controller #(
  parameter NUM_CONTROLS = 1,
  parameter NUM_LOADS    = 1,
  parameter NUM_STORES   = 1,
  parameter DATA_TYPE   = 32,
  parameter ADDR_TYPE   = 32
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
  // Load address input channels
  input  [ (NUM_LOADS * ADDR_TYPE) - 1 : 0] ldAddr,
  input  [                NUM_LOADS - 1 : 0] ldAddr_valid,
  output [                NUM_LOADS - 1 : 0] ldAddr_ready,
  // Load data output channels
  output [ (NUM_LOADS * DATA_TYPE) - 1 : 0] ldData,
  output [                NUM_LOADS - 1 : 0] ldData_valid,
  input  [                NUM_LOADS - 1 : 0] ldData_ready,
  // Store Address Input Channels
  input  [(NUM_STORES * ADDR_TYPE) - 1 : 0] stAddr,
  input  [               NUM_STORES - 1 : 0] stAddr_valid,
  output [               NUM_STORES - 1 : 0] stAddr_ready,
  // Store Data Input Channels
  input  [(NUM_STORES * DATA_TYPE) - 1 : 0] stData,
  input  [               NUM_STORES - 1 : 0] stData_valid,
  output [               NUM_STORES - 1 : 0] stData_ready,
  // Interface to Dual-port BRAM
  input  [               DATA_TYPE - 1 : 0] loadData,
  output                                     loadEn,
  output [               ADDR_TYPE - 1 : 0] loadAddr,
  output                                     storeEn,
  output [               ADDR_TYPE - 1 : 0] storeAddr,
  output [               DATA_TYPE - 1 : 0] storeData
);
  // Internal signal declarations
  wire                      dropLoadEn;
  wire [ADDR_TYPE - 1 : 0] dropLoadAddr;
  wire [DATA_TYPE - 1 : 0] dropLoadData;

  mem_controller_loadless #(
    .NUM_CONTROLS(NUM_CONTROLS),
    .NUM_STORES  (NUM_STORES),
    .DATA_TYPE  (DATA_TYPE),
    .ADDR_TYPE  (ADDR_TYPE)
  ) stores (
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
    .ARBITER_SIZE(NUM_LOADS),
    .ADDR_TYPE  (ADDR_TYPE),
    .DATA_TYPE  (DATA_TYPE)
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
