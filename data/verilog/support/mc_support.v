`timescale 1ns / 1ps

//
//  read_address_mux Module
//

module read_address_mux #(
  parameter ARBITER_SIZE = 1,
  parameter ADDR_TYPE   = 32
) (
  input  [             ARBITER_SIZE - 1 : 0] sel,
  input  [ARBITER_SIZE * ADDR_TYPE - 1 : 0] addr_in,
  output [               ADDR_TYPE - 1 : 0] addr_out
);
  integer                      i;
  reg     [ADDR_TYPE - 1 : 0] addr_out_var;

  always @(*) begin
    addr_out_var = {ADDR_TYPE{1'b0}};

    for (i = 0; i < ARBITER_SIZE; i = i + 1) begin
      if (sel[i]) begin
        addr_out_var = addr_in[i*ADDR_TYPE+:ADDR_TYPE];
      end
    end
  end

  assign addr_out = addr_out_var;

endmodule

//
//  read_address_ready Module
//

module read_address_ready #(
  parameter ARBITER_SIZE = 1
) (
  input  [ARBITER_SIZE-1:0] sel,
  input  [ARBITER_SIZE-1:0] nReady,
  output [ARBITER_SIZE-1:0] ready
);
  assign ready = nReady & sel;

endmodule

//
//  read_data_signals Module
//

module read_data_signals #(
  parameter ARBITER_SIZE = 1,
  parameter DATA_TYPE   = 32
) (
  input                                          rst,
  input                                          clk,
  input      [             ARBITER_SIZE - 1 : 0] sel,
  input      [               DATA_TYPE - 1 : 0] read_data,
  output reg [ARBITER_SIZE * DATA_TYPE - 1 : 0] out_data,
  output reg [             ARBITER_SIZE - 1 : 0] valid,
  input      [             ARBITER_SIZE - 1 : 0] nReady
);
  reg     [              ARBITER_SIZE - 1 : 0] sel_prev = 0;
  reg     [ARBITER_SIZE * DATA_TYPE  - 1 : 0] out_reg = 0;

  integer                                      i;
  initial valid = 0;

  always @(posedge clk) begin
    if (rst) begin
      valid    <= 0;
      sel_prev <= 0;
    end else begin
      sel_prev <= sel;
      for (i = 0; i <= ARBITER_SIZE - 1; i = i + 1)
      if (sel[i]) valid[i] <= 1;
      else if (nReady[i]) valid[i] <= 0;
    end
  end

  always @(posedge clk) begin
    for (i = 0; i <= ARBITER_SIZE - 1; i = i + 1)
      if (rst)
        out_reg[i*DATA_TYPE+:DATA_TYPE] <= 0;
      else if (sel_prev[i])
        out_reg[i*DATA_TYPE+:DATA_TYPE] <= read_data;
    end

  always @(*) begin
    for (i = 0; i <= ARBITER_SIZE - 1; i = i + 1) begin
      if (sel_prev[i]) out_data[i*DATA_TYPE+:DATA_TYPE] = read_data;
      else out_data[i*DATA_TYPE+:DATA_TYPE] = out_reg[i*DATA_TYPE+:DATA_TYPE];
    end
  end

endmodule

//
//  read_priority Module
//

module read_priority #(
  parameter ARBITER_SIZE = 1
) (
  input      [ARBITER_SIZE - 1 : 0] req,
  input      [ARBITER_SIZE - 1 : 0] data_ready,
  output reg [ARBITER_SIZE - 1 : 0] priority_out
);
  reg     prio_req = 0;
  integer i;

  always @(req, data_ready) begin
    priority_out[0] = req[0] & data_ready[0];
    prio_req        = req[0] & data_ready[0];

    for (i = 1; i <= ARBITER_SIZE - 1; i = i + 1) begin
      priority_out[i] = ~prio_req & req[i] & data_ready[i];
      prio_req        = prio_req | (req[i] & data_ready[i]);
    end
  end
endmodule

module read_memory_arbiter #(
  parameter ARBITER_SIZE = 2,
  parameter ADDR_TYPE   = 32,
  parameter DATA_TYPE   = 32
) (
  input                                      clk,
  input                                      rst,
  // Interface to previous
  input  [             ARBITER_SIZE - 1 : 0] pValid,
  output [             ARBITER_SIZE - 1 : 0] ready,
  input  [ARBITER_SIZE * ADDR_TYPE - 1 : 0] address_in,
  // Interface to next
  input  [             ARBITER_SIZE - 1 : 0] nReady,
  output [             ARBITER_SIZE - 1 : 0] valid,
  output [ARBITER_SIZE * DATA_TYPE - 1 : 0] data_out,
  // Interface to memory
  output                                     read_enable,
  output [               ADDR_TYPE - 1 : 0] read_address,
  input  [               DATA_TYPE - 1 : 0] data_from_memory
);

  wire [ARBITER_SIZE - 1 : 0] priorityOut;

  // Instance of read_priority
  read_priority #(
    .ARBITER_SIZE(ARBITER_SIZE)
  ) prio (
    .req         (pValid),
    .data_ready  (nReady),
    .priority_out(priorityOut)
  );

  // Instance of read_address_mux
  read_address_mux #(
    .ARBITER_SIZE(ARBITER_SIZE),
    .ADDR_TYPE  (ADDR_TYPE)
  ) addressing (
    .sel     (priorityOut),
    .addr_in (address_in),
    .addr_out(read_address)
  );

  // Instance of read_address_ready
  read_address_ready #(
    .ARBITER_SIZE(ARBITER_SIZE)
  ) adderssReady (
    .sel   (priorityOut),
    .nReady(nReady),
    .ready (ready)
  );

  // Instance of read_data_signals
  read_data_signals #(
    .ARBITER_SIZE(ARBITER_SIZE),
    .DATA_TYPE  (DATA_TYPE)
  ) data_signals_inst (
    .rst      (rst),
    .clk      (clk),
    .sel      (priorityOut),
    .read_data(data_from_memory),
    .out_data (data_out),
    .valid    (valid),
    .nReady   (nReady)
  );

  assign read_enable = |priorityOut;

endmodule


//
//  write_address_mux Module
//

module write_address_mux #(
  parameter ARBITER_SIZE = 1,
  parameter ADDR_TYPE   = 32
) (
  input  [             ARBITER_SIZE - 1 : 0] sel,
  input  [ARBITER_SIZE * ADDR_TYPE - 1 : 0] addr_in,
  output [               ADDR_TYPE - 1 : 0] addr_out
);
  integer                      i;
  reg     [ADDR_TYPE - 1 : 0] addr_out_var;

  always @(*) begin
    //! May need to chaneg the following line to remove the assignment to 0
    addr_out_var = {ADDR_TYPE{1'b0}};

    for (i = 0; i < ARBITER_SIZE; i = i + 1) begin
      if (sel[i]) begin
        addr_out_var = addr_in[i*ADDR_TYPE+:ADDR_TYPE];
      end
    end
  end

  assign addr_out = addr_out_var;

endmodule

//
//  write_address_mux Module
//

module write_address_ready #(
  parameter ARBITER_SIZE = 1
) (
  input  [ARBITER_SIZE-1:0] sel,
  input  [ARBITER_SIZE-1:0] nReady,
  output [ARBITER_SIZE-1:0] ready
);
  assign ready = nReady & sel;

endmodule

//
//  write_data_signals Module
//

module write_data_signals #(
  parameter ARBITER_SIZE = 1,
  parameter DATA_TYPE   = 32
) (
  input                                           clk,
  input                                           rst,
  input      [              ARBITER_SIZE - 1 : 0] sel,
  output     [                DATA_TYPE - 1 : 0] write_data,
  input      [ARBITER_SIZE * DATA_TYPE  - 1 : 0] in_data,
  output     [              ARBITER_SIZE - 1 : 0] valid
);
  integer                      i;
  reg     [DATA_TYPE - 1 : 0] data_out_var = 0;
  reg     [ARBITER_SIZE - 1: 0] valid_out_var = 0;

  always @(*) begin
    data_out_var = {DATA_TYPE{1'b0}};
    for (i = 0; i < ARBITER_SIZE; i = i + 1) begin
      if (sel[i]) begin
        data_out_var = in_data[i*DATA_TYPE+:DATA_TYPE];
      end
    end
  end

  assign write_data = data_out_var;

  always @(*) begin
    if (rst) begin
      valid_out_var <= 0;
    end else begin
      valid_out_var <= sel;
    end
  end

  assign valid = valid_out_var;

endmodule

//
//  write_priority Module
//

module write_priority #(
  parameter ARBITER_SIZE = 1
) (
  input      [ARBITER_SIZE - 1 : 0] req,
  input      [ARBITER_SIZE - 1 : 0] data_ready,
  output reg [ARBITER_SIZE - 1 : 0] priority_out
);
  reg     prio_req = 0;
  integer i;

  always @(req, data_ready) begin
    priority_out[0] = req[0] & data_ready[0];
    prio_req        = req[0] & data_ready[0];

    for (i = 1; i <= ARBITER_SIZE - 1; i = i + 1) begin
      priority_out[i] = ~prio_req & req[i] & data_ready[i];
      prio_req        = prio_req | (req[i] & data_ready[i]);
    end
  end

endmodule

//
//  write_memory_arbiter Module
//

module write_memory_arbiter #(
  parameter ARBITER_SIZE = 2,
  parameter ADDR_TYPE   = 32,
  parameter DATA_TYPE   = 32
) (
  input                                      rst,
  input                                      clk,
  // Interface to previous
  input  [             ARBITER_SIZE - 1 : 0] pValid,
  output [             ARBITER_SIZE - 1 : 0] ready,
  input  [ADDR_TYPE * ARBITER_SIZE - 1 : 0] address_in,
  input  [DATA_TYPE * ARBITER_SIZE - 1 : 0] data_in,
  // Interface to next    
  input  [             ARBITER_SIZE - 1 : 0] nReady,
  output [             ARBITER_SIZE - 1 : 0] valid,
  // Interface to memory   
  output                                     write_enable,
  output [               ADDR_TYPE - 1 : 0] write_address,
  output [               DATA_TYPE - 1 : 0] data_to_memory
);
  wire [ARBITER_SIZE - 1 : 0] priorityOut;

  // Priority handling
  write_priority #(
    .ARBITER_SIZE(ARBITER_SIZE)
  ) prio (
    .req         (pValid),
    .data_ready  (nReady),
    .priority_out(priorityOut)
  );

  // Address multiplexing
  write_address_mux #(
    .ARBITER_SIZE(ARBITER_SIZE),
    .ADDR_TYPE  (ADDR_TYPE)
  ) addressing (
    .sel     (priorityOut),
    .addr_in (address_in),
    .addr_out(write_address)
  );

  // Ready signal handling
  write_address_ready #(
    .ARBITER_SIZE(ARBITER_SIZE)
  ) addressReady (
    .sel   (priorityOut),
    .nReady(nReady),
    .ready (ready)
  );

  // Data handling
  write_data_signals #(
    .ARBITER_SIZE(ARBITER_SIZE),
    .DATA_TYPE  (DATA_TYPE)
  ) data_signals_inst (
    .rst       (rst),
    .clk       (clk),
    .sel       (priorityOut),
    .in_data   (data_in),
    .write_data(data_to_memory),
    .valid     (valid)
  );

  assign write_enable = |priorityOut;

endmodule

//
//  mc_control
//

module mc_control (
  input  clk,
  input  rst,
  // start input control
  input  memStart_valid,
  output memStart_ready,
  // end output control
  output memEnd_valid,
  input  memEnd_ready,
  // "no more requests" input control
  input  ctrlEnd_valid,
  output ctrlEnd_ready,
  // all requests completed
  input  allRequestsDone
);
  reg memIdle = 1;
  reg memDone = 0;
  reg memAckCtrl = 0;

  assign memStart_ready = memIdle;
  assign memEnd_valid   = memDone;
  assign ctrlEnd_ready  = memAckCtrl;

  always @(posedge clk) begin
    if (rst) begin
      memIdle    <= 1;
      memDone    <= 0;
      memAckCtrl <= 0;
    end else begin
      memIdle    <= memIdle;
      memDone    <= memDone;
      memAckCtrl <= memAckCtrl;

      // determine when the memory has completed all requests
      if (ctrlEnd_valid && allRequestsDone) begin
        memDone    <= 1;
        memAckCtrl <= 1;
      end

      // acknowledge the 'ctrlEnd' control
      if (ctrlEnd_valid && memAckCtrl) begin
        memAckCtrl <= 0;
      end

      // determine when the memory is idle
      if (memStart_valid && memIdle) begin
        memIdle <= 0;
      end
      if (memDone && memEnd_ready) begin
        memIdle <= 1;
        memDone <= 0;
      end
    end
  end
endmodule


//
//  burst support modules
//

// burst_counter Module

module burst_counter #(
  parameter BURST_TYPE = 16
) (
  input                         clk,
  input                         rst,
  input                         read_enable,
  input                         past_active_transaction,
  input     [BURST_TYPE - 1 :0] burst_length,
  output                        active_transaction
);
  reg [BURST_TYPE - 1 : 0] burst_count;

  always @(posedge clk) begin
    if (rst) begin
      burst_count <= 0;
    end else begin
      if (read_enable && !past_active_transaction) begin
        // start of a new burst transaction
        burst_count <= burst_length;
      end else if (past_active_transaction) begin
        // continue burst transaction
        burst_count <= burst_count - 1;
      end
    end
  end

  assign active_transaction = burst_count > 1 || (read_enable && !past_active_transaction);

endmodule


// burst_manager Module

module burst_manager #(
  parameter ARBITER_SIZE = 2,
  parameter BURST_TYPE = 16
) (
  input                         clk,
  input                         rst,
  input [ARBITER_SIZE * BURST_TYPE - 1 :0]     burst_length_in,
  input [ARBITER_SIZE - 1 :0]   pValid_burst,
  input                         any_valid_request,
  input [ARBITER_SIZE - 1 :0]   tmp_priorityOut,
  output reg [ARBITER_SIZE - 1 :0] priorityOut,
  output reg                    active_transaction,
  output reg                    new_transaction,
  output reg [BURST_TYPE - 1 :0] burst_length_value,
  output reg [ARBITER_SIZE - 1 : 0] burst_length_ready
);

  reg [BURST_TYPE - 1 : 0] burst_count;
  integer i;

  always @(posedge clk) begin
    if (rst) begin
      burst_count          <= 0;
      active_transaction   <= 0;
      priorityOut          <= 0;
      burst_length_value   <= 0;
      burst_length_ready   <= 0;
      new_transaction      <= 0;
    end else begin
      burst_length_ready   <= 0;
      new_transaction      <= 0;
      if (!active_transaction && any_valid_request) begin
        // Start new burst
        for (i = 0; i < ARBITER_SIZE; i = i + 1) begin
          if (pValid_burst[i] && tmp_priorityOut[i]) begin
            burst_count <= burst_length_in[i*BURST_TYPE +: BURST_TYPE];
            burst_length_value <= burst_length_in[i*BURST_TYPE +: BURST_TYPE];
            priorityOut      <= tmp_priorityOut;
            active_transaction <= 1;
            burst_length_ready[i] <= 1;
            new_transaction <= 1;
          end
        end
      end
      else if (active_transaction) begin
        // End of burst
        if (burst_count <= 1) begin
          active_transaction <= 0;
          priorityOut <= 0;
        end
        // Decrement burst counter
        if (burst_count > 0)
          burst_count <= burst_count - 1;
      end
    end
  end

endmodule


//
//  read_address_burst_ready Module
//

module read_address_burst_ready #(
  parameter ARBITER_SIZE = 1
) (
  input  [ARBITER_SIZE-1:0] sel,
  input  [ARBITER_SIZE-1:0] nReady,
  input  new_transaction,
  output [ARBITER_SIZE-1:0] ready
);
  assign ready = nReady & sel & {ARBITER_SIZE{new_transaction}};

endmodule


//  read burst arbiter


module read_memory_arbiter_burst #(
  parameter ARBITER_SIZE = 2,
  parameter ADDR_TYPE   = 32,
  parameter DATA_TYPE   = 32,
  parameter BURST_TYPE   = 16
) (
  input                                      clk,
  input                                      rst,
  //// Interface to previous
  // address
  input  [             ARBITER_SIZE - 1 : 0] pValid,
  output [             ARBITER_SIZE - 1 : 0] ready,
  input  [ARBITER_SIZE * ADDR_TYPE - 1 : 0] address_in,
  // burst length
  input  [             ARBITER_SIZE - 1 : 0] pValid_burst,
  output [             ARBITER_SIZE - 1 : 0] ready_burst,
  input  [ARBITER_SIZE * BURST_TYPE - 1 : 0] burst_length_in,
  //// Interface to next
  input  [             ARBITER_SIZE - 1 : 0] nReady,
  output [             ARBITER_SIZE - 1 : 0] valid,
  output [ARBITER_SIZE * DATA_TYPE - 1 : 0] data_out,
  //// Interface to memory
  output                                     read_enable,
  output [               ADDR_TYPE - 1 : 0] read_address,
  output [               BURST_TYPE - 1 : 0] read_burst_length,
  input  [               DATA_TYPE - 1 : 0] data_from_memory
);

  wire [ARBITER_SIZE - 1 : 0] tmp_priorityOut;
  wire [ARBITER_SIZE - 1 : 0] priorityOut;
  wire  active_transaction;
  wire tmp_valid;
  wire any_valid_request;
  wire new_transaction;

  // Instance of read_priority
  read_priority #(
    .ARBITER_SIZE(ARBITER_SIZE)
  ) prio (
    .req         (pValid),
    .data_ready  (nReady),
    .priority_out(tmp_priorityOut)
  );

  // Instance of read_address_mux
  read_address_mux #(
    .ARBITER_SIZE(ARBITER_SIZE),
    .ADDR_TYPE  (ADDR_TYPE)
  ) addressing (
    .sel     (priorityOut),
    .addr_in (address_in),
    .addr_out(read_address)
  );

  // Instance of read_address_burst_ready
  read_address_burst_ready #(
    .ARBITER_SIZE(ARBITER_SIZE)
  ) addressReady (
    .sel   (tmp_priorityOut),
    .nReady(nReady),
    .new_transaction(new_transaction),
    .ready (ready)
  );

  // Instance of read_data_signals
  read_data_signals #(
    .ARBITER_SIZE(ARBITER_SIZE),
    .DATA_TYPE  (DATA_TYPE)
  ) data_signals_inst (
    .rst      (rst),
    .clk      (clk),
    .sel      (priorityOut),
    .read_data(data_from_memory),
    .out_data (data_out),
    .valid    (valid),
    .nReady   (nReady)
  );

  // Instance of burst management
  burst_manager #(
    .ARBITER_SIZE(ARBITER_SIZE),
    .BURST_TYPE (BURST_TYPE)
  ) burst_mgr (
    .clk                 (clk),
    .rst                 (rst),
    .burst_length_in     (burst_length_in),
    .pValid_burst       (pValid_burst),
    .any_valid_request   (any_valid_request),
    .tmp_priorityOut     (tmp_priorityOut),
    .priorityOut         (priorityOut),
    .active_transaction  (active_transaction),
    .new_transaction      (new_transaction),
    .burst_length_value  (read_burst_length),
    .burst_length_ready  (ready_burst)
  );

  assign read_enable = active_transaction;
  assign any_valid_request = |tmp_priorityOut;

endmodule
