`timescale 1ns/1ps
module elastic_fifo_inner #(
  parameter DATA_TYPE = 32,
  parameter NUM_SLOTS = 2
) (
  input  clk,
  input  rst,
  input  [DATA_TYPE - 1 : 0] ins,
  input  ins_valid,
  input  outs_ready,

  output [DATA_TYPE - 1 : 0] outs,
  output outs_valid,
  output ins_ready
);
  // Internal Signal Definition
  wire ReadEn, WriteEn;
  reg [$clog2(NUM_SLOTS) - 1 : 0] Tail = 0, Head = 0;
  reg Full = 0, Empty = 0;
  reg [DATA_TYPE - 1 : 0] Memory[0 : NUM_SLOTS - 1];
  integer i;
  
  // Ready if there is space in the FIFO
  assign ins_ready = ~Full | outs_ready;

  // Read if next can accept and there is sth in FIFO to read
  assign ReadEn = (outs_ready & ~Empty);
  assign outs_valid = ~Empty;
  assign WriteEn = ins_valid & (~Full | outs_ready);
  assign outs = Memory[Head];

  // Initialize memory content
  initial begin
     for (i=0; i<NUM_SLOTS; i=i+1) begin
        Memory[i] = 0;
     end
  end

  always @(posedge clk) begin
    if (rst) begin
      
    end else if (WriteEn) begin
      Memory[Tail] <= ins;
    end
  end

  // Update Tail
  always @(posedge clk) begin
    if (rst) begin
      Tail <= 0;
    end else begin
      if (WriteEn) begin
        Tail <= (Tail + 1) % NUM_SLOTS;
      end
    end  
  end

  // Update Head
  always @(posedge clk) begin
    if (rst) begin
      Head <= 0;
    end else begin
      if (ReadEn) begin
        Head <= (Head + 1) % NUM_SLOTS;
      end
    end 
  end

  // Update Full
  always @(posedge clk) begin
    if (rst) begin
      Full <= 0;
    end else begin
      // If only filling but not emptying
      if (WriteEn & ~ReadEn) begin
        // If new tail index will reach head index
        if ((Tail + 1) % NUM_SLOTS == Head) begin
          Full <= 1;
        end
      end else if (~WriteEn & ReadEn) begin
        // if only emptying but not filling
        Full <= 0;
      end
    end
  end

  // Update Empty
  always @(posedge clk) begin
    if (rst) begin
      Empty <= 1;
    end else begin
      // If only emptying but not filling
      if (~WriteEn & ReadEn) begin
        if ((Head + 1) % NUM_SLOTS == Tail) begin
          Empty <= 1;
        end
      end else if (WriteEn & ~ReadEn) begin
        // If only filling but not emptying
        Empty <= 0;
      end
    end
  end

endmodule
