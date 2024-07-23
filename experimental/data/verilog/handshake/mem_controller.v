module mem_controller #(
  parameter NUM_CONTROL  = 1,
  parameter NUM_LOAD  = 1,
  parameter NUM_STORE = 1,
  parameter DATA_WIDTH = 32,
  parameter ADDR_WIDTH = 32
)(
  input  clk,
  input  rst,
  // Control input channels
  input  [(NUM_CONTROL * 32) - 1 : 0] ctrl,
  input  [NUM_CONTROL - 1 : 0] ctrl_valid,
  output [NUM_CONTROL - 1 : 0] ctrl_ready,
  // Load address input channels
  input  [(NUM_LOAD * ADDR_WIDTH) - 1 : 0] ldAddr,
  input  [NUM_LOAD - 1 : 0] ldAddr_valid,
  output [NUM_LOAD - 1 : 0] ldAddr_ready,
  // Load data output channels
  output [(NUM_LOAD * DATA_WIDTH) - 1 : 0] ldData,
  output [NUM_LOAD - 1 : 0] ldData_valid,
  input  [NUM_LOAD - 1 : 0] ldData_ready,
  // Store address input channels
  input  [(NUM_STORE * ADDR_WIDTH) - 1 : 0] stAddr,
  input  [NUM_STORE - 1 : 0] stAddr_valid,
  output [NUM_STORE - 1 : 0] stAddr_ready,
  // Store data input channels
  input  [(NUM_STORE * DATA_WIDTH) - 1 : 0] stData,
  input  [NUM_STORE - 1 : 0] stData_valid,
  output [NUM_STORE - 1 : 0] stData_ready,
  // Memory done channel
  output memDone_valid,
  input  memDone_ready,
  // Interface to dual-port BRAM
  input  [DATA_WIDTH - 1 : 0] loadData,
  output loadEn,
  output [ADDR_WIDTH - 1 : 0] loadAddr,
  output storeEn,
  output [ADDR_WIDTH - 1 : 0] storeAddr,
  output [DATA_WIDTH - 1 : 0] storeData
);
  // Internal signal declarations
  wire dropLoadEn;
  wire [ADDR_WIDTH - 1 : 0] dropLoadAddr;
  wire [DATA_WIDTH - 1 : 0] dropLoadData;

  mem_controller_loadless #(
    .NUM_CONTROL  (NUM_CONTROL ),
    .NUM_STORE (NUM_STORE),
    .DATA_WIDTH  (DATA_WIDTH ),
    .ADDR_WIDTH  (ADDR_WIDTH )
  ) stores (
    .clk           (clk          ),
    .rst           (rst          ),
    .ctrl          (ctrl         ),
    .ctrl_valid    (ctrl_valid   ),
    .ctrl_ready    (ctrl_ready   ),
    .stAddr        (stAddr       ),
    .stAddr_valid  (stAddr_valid ),
    .stAddr_ready  (stAddr_ready ),
    .stData        (stData       ),
    .stData_valid  (stData_valid ),
    .stData_ready  (stData_ready ),
    .memDone_valid (memDone_valid),
    .memDone_ready (memDone_ready),
    .loadData      (dropLoadData ),
    .loadEn        (dropLoadEn   ),
    .loadAddr      (dropLoadAddr ),
    .storeEn       (storeEn      ),
    .storeAddr     (storeAddr    ),
    .storeData     (storeData    )
  );

  read_memory_arbiter #(
    .ARBITER_SIZE (NUM_LOAD),
    .ADDR_WIDTH   (ADDR_WIDTH),
    .DATA_WIDTH   (DATA_WIDTH)
  ) read_arbiter (
    .rst              (rst         ),
    .clk              (clk         ),
    .pValid           (ldAddr_valid),
    .ready            (ldAddr_ready),
    .address_in       (ldAddr      ),
    .nReady           (ldData_ready),
    .valid            (ldData_valid),
    .data_out         (ldData      ),
    .read_enable      (loadEn      ),
    .read_address     (loadAddr    ),
    .data_from_memory (loadData    )
  );

endmodule
