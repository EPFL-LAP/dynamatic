module mem_controller_loadless #(
  parameter CTRL_COUNT = 1,
  parameter STORE_COUNT = 1,
  parameter DATA_WIDTH = 32,
  parameter ADDR_WIDTH = 32
)(
  input  clk,
  input  rst,
  // Control Input Channels
  input  [(CTRL_COUNT * 32) - 1 : 0] ctrl,
  input  [CTRL_COUNT - 1 : 0] ctrl_valid,
  output [CTRL_COUNT - 1 : 0] ctrl_ready,
  // Store Address Input Channels
  input  [(STORE_COUNT * ADDR_WIDTH) - 1 : 0] stAddr,
  input  [STORE_COUNT - 1 : 0] stAddr_valid,
  output [STORE_COUNT - 1 : 0] stAddr_ready,
  // Store Data Input Channels
  input  [(STORE_COUNT * DATA_WIDTH) - 1 : 0] stData,
  input  [STORE_COUNT - 1 : 0] stData_valid,
  output [STORE_COUNT - 1 : 0] stData_ready,
  // Memory Done Channel
  output memDone_valid,
  input  memDone_ready,
  // Interface to Dual-port BRAM
  input  [DATA_WIDTH - 1 : 0] loadData,
  output loadEn,
  output [ADDR_WIDTH - 1 : 0] loadAddr,
  output storeEn,
  output [ADDR_WIDTH - 1 : 0] storeAddr,
  output [DATA_WIDTH - 1 : 0] storeData
);
  // Internal Signals
  reg remainingStores = 0;
  wire [STORE_COUNT - 1 : 0] storePorts_valid, storePorts_ready;

  assign loadEn = 0;
  assign loadAddr = {ADDR_WIDTH{1'b0}};

  // Instantiate write memory arbiter
  write_memory_arbiter #(
    .ARBITER_SIZE (STORE_COUNT),
    .ADDR_WIDTH   (ADDR_WIDTH),
    .DATA_WIDTH   (DATA_WIDTH)
  ) write_arbiter (
    .rst            (rst                ),
    .clk            (clk                ),
    .pValid         (stAddr_valid       ),
    .ready          (storePorts_ready   ),
    .address_in     (stAddr             ),
    .data_in        (stData             ),
    .nReady         ({STORE_COUNT{1'b1}}),
    .valid          (storePorts_valid   ),
    .write_enable   (storeEn            ),
    .write_address  (storeAddr          ),
    .data_to_memory (storeData          )
  );

  assign stData_ready = storePorts_ready;
  assign stAddr_ready = storePorts_ready;

  // Counting Stores
  always @(posedge clk, posedge rst) begin
    if (rst) begin
      remainingStores <= 32'd0;
    end else begin
      integer i;
      for (i = 0; i < CTRL_COUNT; i = i + 1) begin
        if (ctrl_valid[i]) begin
          remainingStores <= remainingStores + ctrl[(i * 32) + 31 -: 32];
        end
      end
      if (storeEn) begin
        remainingStores <= remainingStores - 1;
      end
    end
  end

  // Memory Done logic
  assign memDone_valid = (~(|remainingStores)) & (~(|ctrl_valid));
  assign ctrl_ready = {CTRL_COUNT{1'b1}};

endmodule