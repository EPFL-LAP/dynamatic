`timescale 1ns / 1ps
module mem_controller_li #(
  parameter NUM_CONTROLS = 1,
  parameter NUM_BURST_LOADS    = 1,
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
  // Load address input channels
  input  [ (NUM_BURST_LOADS * ADDR_TYPE) - 1 : 0] ldAddr,
  input  [                NUM_BURST_LOADS - 1 : 0] ldAddr_valid,
  output [                NUM_BURST_LOADS - 1 : 0] ldAddr_ready,
  // Load burst length input channels
  input  [(NUM_BURST_LOADS * BURST_TYPE) - 1 : 0] loadBurstLength,
  input  [               NUM_BURST_LOADS - 1 : 0] loadBurstLength_valid,
  output [               NUM_BURST_LOADS - 1 : 0] loadBurstLength_ready,
  // Load data output channels
  output [ (NUM_BURST_LOADS * DATA_TYPE) - 1 : 0] ldData,
  output [                NUM_BURST_LOADS - 1 : 0] ldData_valid,
  input  [                NUM_BURST_LOADS - 1 : 0] ldData_ready,
  // Store Address Input Channels
  input  [(NUM_STORES * ADDR_TYPE) - 1 : 0] stAddr,
  input  [               NUM_STORES - 1 : 0] stAddr_valid,
  output [               NUM_STORES - 1 : 0] stAddr_ready,
  // Store Data Input Channels
  input  [(NUM_STORES * DATA_TYPE) - 1 : 0] stData,
  input  [               NUM_STORES - 1 : 0] stData_valid,
  output [               NUM_STORES - 1 : 0] stData_ready,
  // Interface to Dual-port BRAM
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

  // Unused wires from loadless store controller
  // Load data to be dropped
  wire [DATA_TYPE - 1 : 0] dropLoadData;
  wire  dropLoadData_valid;
  wire  dropLoadData_ready;
  // Load address to be dropped
  wire [ADDR_TYPE - 1 : 0] dropLoadAddr;
  wire dropLoadAddr_valid;
  wire dropLoadAddr_ready;
  wire [BURST_TYPE - 1 : 0] dropLoadBurstLen;

  assign loadBurstLen = {BURST_TYPE{1'b1}}; // Max burst length

  mem_controller_li_loadless #(
    .NUM_CONTROLS(NUM_CONTROLS),
    .NUM_STORES  (NUM_STORES),
    .DATA_TYPE  (DATA_TYPE),
    .ADDR_TYPE  (ADDR_TYPE),
    .BURST_TYPE (BURST_TYPE)
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
    .loadData_valid(dropLoadData_valid),
    .loadData_ready(dropLoadData_ready),
    .loadAddr      (dropLoadAddr),
    .loadAddr_valid(dropLoadAddr_valid),
    .loadAddr_ready(dropLoadAddr_ready),
    .loadBurstLen  (dropLoadBurstLen),
    .storeAddr     (storeAddr),
    .storeAddr_valid(storeAddr_valid),
    .storeAddr_ready(storeAddr_ready),
    .storeBurstLen (storeBurstLen),
    .storeData     (storeData),
    .storeData_valid(storeData_valid),
    .storeData_ready(storeData_ready)
  );

  read_li_memory_arbiter #(
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
    .ready_data      (loadData_ready),
    .data_from_memory (loadData),
    .pValid_data     (loadData_valid),
    .nReady_address  (loadAddr_ready),
    .valid_address   (loadAddr_valid),
    .read_address    (loadAddr)
  );

endmodule
