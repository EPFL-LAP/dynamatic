module elastic_fifo_inner #(
  parameter SIZE = 2,
  parameter DATA_WIDTH = 32
) (
  input  clk,
  input  rst,
  input  [DATA_WIDTH - 1 : 0] ins,
  input  ins_valid,
  input  outs_ready,

  output [DATA_WIDTH - 1 : 0] outs,
  output outs_valid,
  output ins_ready
);
  // Internal Signal Definition
  wire ReadEn, WriteEn;
  reg [$clog2(SIZE) - 1 : 0] Tail = 0, Head = 0;
  reg Full = 0, Empty = 0, fifo_valid;

  reg [DATA_WIDTH - 1 : 0] Memory[0 : SIZE - 1];
  
  // Ready if there is space in the FIFO
  assign ins_ready = ~Full | outs_ready;

  // Read if next can accept and there is sth in FIFO to read
  assign ReadEn = (outs_ready & ~Empty);
  assign outs_valid = ~Empty;
  assign WriteEn = ins_valid & (~Full | outs_ready);
  assign outs = Memory[Head];

  // Update FIFO valid
  always @(posedge clk, posedge rst) begin
    if (rst) begin
      fifo_valid <= 0;
    end else if (ReadEn) begin
      fifo_valid <= 1;
    end else if (outs_ready) begin
      fifo_valid <= 0;
    end
  end

  always @(posedge clk, posedge rst) begin
    if (rst) begin
      
    end else if (WriteEn) begin
      Memory[Tail] <= ins;
    end
  end

  // Update Tail
  always @(posedge clk, posedge rst) begin
    if (rst) begin
      Tail <= 0;
    end else begin
      if (WriteEn) begin
        Tail <= (Tail + 1) % SIZE;
      end
    end  
  end

  // Update Head
  always @(posedge clk, posedge rst) begin
    if (rst) begin
      Head <= 0;
    end else begin
      if (ReadEn) begin
        Head <= (Head + 1) % SIZE;
      end
    end 
  end

  // Update Full
  always @(posedge clk, posedge rst) begin
    if (rst) begin
      Full <= 0;
    end else begin
      // If only filling but not emptying
      if (WriteEn & ~ReadEn) begin
        // If new tail index will reach head index
        if ((Tail + 1) % SIZE == Head) begin
          Full <= 1;
        end
      end else if (~WriteEn & ReadEn) begin
        // if only emptying but not filling
        Full <= 0;
      end
    end
  end

  // Update Empty
  always @(posedge clk, posedge rst) begin
    if (rst) begin
      Empty <= 1;
    end else begin
      // If only emptying but not filling
      if (~WriteEn & ReadEn) begin
        if ((Head + 1) % SIZE == Tail) begin
          Empty <= 1;
        end
      end else if (WriteEn & ~ReadEn) begin
        // If only filling but not emptying
        Empty <= 0;
      end
    end
  end

endmodule
