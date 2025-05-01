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

  always @(posedge clk) begin
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
  output                                     enable,
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
  assign enable       = |priorityOut;

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
