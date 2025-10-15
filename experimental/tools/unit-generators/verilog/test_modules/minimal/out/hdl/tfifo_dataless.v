// dataless_tfifo : dataless_tfifo({'data_type': 32, 'num_slots': 2})

`timescale 1ns/1ps

`timescale 1ns/1ps

// dataless Elastic FIFO Inner Module

module elastic_fifo_inner_dataless #(
  parameter NUM_SLOTS = 2
)(
  input  clk,
  input  rst,
  input  ins_valid,
  input  outs_ready,

  output ins_ready,
  output outs_valid
);
  wire ReadEn, WriteEn;
  reg [$clog2(NUM_SLOTS) - 1 : 0] Tail = 0, Head = 0;
  reg Full = 0, Empty = 1;

  // Ready if there is space in the FIFO
  assign ins_ready = ~Full | outs_ready;

  // Read if next can accept and there is sth in FIFO to read
  assign ReadEn = (outs_ready & ~Empty);
  assign outs_valid = ~Empty;
  assign WriteEn = ins_valid & (~Full | outs_ready);

  // Update Tail
  always @(posedge clk) begin
    if (rst) begin
      Tail <= 0;
    end else begin
      if (WriteEn) begin
        Tail <= (Tail + 1 == NUM_SLOTS) ? 0 : Tail + 1;
      end
    end  
  end

  // Update Head
  always @(posedge clk) begin
    if (rst) begin
      Head <= 0;
    end else begin
      if (ReadEn) begin
        Head <= (Head + 1 == NUM_SLOTS) ? 0 : Head + 1;
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
        if (((Tail + 1 == NUM_SLOTS) ? 0 : Tail + 1) == Head) begin
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
        if (((Head + 1 == NUM_SLOTS) ? 0 : Head + 1) == Tail) begin
          Empty <= 1;
        end
      end else if (WriteEn & ~ReadEn) begin
        // If only filling but not emptying
        Empty <= 0;
      end
    end
  end

endmodule


// dataless tfifo Module
module dataless_tfifo #(
  parameter NUM_SLOTS = 2
)(
  input  clk,
  input  rst,
  // Input channel
  input  ins_valid,
  output ins_ready,
  // Output channel
  output outs_valid,
  input  outs_ready
);
  wire fifo_valid, fifo_ready;
  wire fifo_pvalid, fifo_nready;

  assign outs_valid = ins_valid || fifo_valid;
  assign ins_ready = fifo_ready || outs_ready;
  assign fifo_pvalid = ins_valid && (!outs_ready || fifo_valid);
  assign fifo_nready = outs_ready;

  elastic_fifo_inner_dataless #(
    .NUM_SLOTS(NUM_SLOTS)
  ) fifo (
    .clk        (clk        ),
    .rst        (rst        ),
    .ins_valid  (fifo_pvalid),
    .outs_ready (fifo_nready),
    .outs_valid (fifo_valid ),
    .ins_ready  (fifo_ready )
  );

endmodule


