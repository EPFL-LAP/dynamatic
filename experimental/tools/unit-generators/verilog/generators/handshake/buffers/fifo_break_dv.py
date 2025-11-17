from generators.handshake.buffers.one_slot_break_dv import generate_one_slot_break_dv


def generate_fifo_break_dv(name, params):
    bitwidth = params["bitwidth"]
    num_slots = params["num_slots"]

    if num_slots == 1:
        return generate_one_slot_break_dv(name, params)
    elif bitwidth == 0:
        return _generate_fifo_break_dv_dataless(name, num_slots)
    else:
        return _generate_fifo_break_dv(name, {"num_slots": num_slots, "bitwidth": bitwidth})


def _generate_fifo_break_dv(name, params):
    num_slots = params["num_slots"]
    bitwidth = params["bitwidth"]

    if (bitwidth == 0):
        return _generate_fifo_break_dv_dataless(name, params)

    return f"""
// Module of fifo_break_dv
module {name}(
  input  clk,
  input  rst,
  input  [{bitwidth} - 1 : 0] ins,
  input  ins_valid,
  input  outs_ready,

  output [{bitwidth} - 1 : 0] outs,
  output outs_valid,
  output ins_ready
);
  // Internal Signal Definition
  wire ReadEn, WriteEn;
  reg [$clog2({num_slots}) - 1 : 0] Tail = 0, Head = 0;
  reg Full = 0, Empty = 1;
  reg [{bitwidth} - 1 : 0] Memory[0 : {num_slots} - 1];
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
     for (i=0; i<{num_slots}; i=i+1) begin
        Memory[i] = 0;
     end
  end

  always @(posedge clk) begin
    if (rst) begin
      // Clear memory on reset to mirror VHDL semantics
      for (i = 0; i < {num_slots}; i = i + 1) begin
        Memory[i] <= {bitwidth}'d0;
      end
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
        Tail <= (Tail + 1 == {num_slots}) ? 0 : Tail + 1;
      end
    end  
  end

  // Update Head
  always @(posedge clk) begin
    if (rst) begin
      Head <= 0;
    end else begin
      if (ReadEn) begin
        Head <= (Head + 1 == {num_slots}) ? 0 : Head + 1;
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
        if (((Tail + 1 == {num_slots}) ? 0 : Tail + 1) == Head) begin
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
        if (((Head + 1 == {num_slots}) ? 0 : Head + 1) == Tail) begin
          Empty <= 1;
        end
      end else if (WriteEn & ~ReadEn) begin
        // If only filling but not emptying
        Empty <= 0;
      end
    end
  end

endmodule
"""


def _generate_fifo_break_dv_dataless(name, params):

    num_slots = params["num_slots"]

    return f"""
// Module of fifo_break_dv_dataless

module {name}(
  input  clk,
  input  rst,
  input  ins_valid,
  input  outs_ready,

  output ins_ready,
  output outs_valid
);
  wire ReadEn, WriteEn;
  reg [$clog2({num_slots}) - 1 : 0] Tail = 0, Head = 0;
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
        Tail <= (Tail + 1 == {num_slots}) ? 0 : Tail + 1;
      end
    end  
  end

  // Update Head
  always @(posedge clk) begin
    if (rst) begin
      Head <= 0;
    end else begin
      if (ReadEn) begin
        Head <= (Head + 1 == {num_slots}) ? 0 : Head + 1;
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
        if (((Tail + 1 == {num_slots}) ? 0 : Tail + 1) == Head) begin
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
        if (((Head + 1 == {num_slots}) ? 0 : Head + 1) == Tail) begin
          Empty <= 1;
        end
      end else if (WriteEn & ~ReadEn) begin
        // If only filling but not emptying
        Empty <= 0;
      end
    end
  end

endmodule
"""